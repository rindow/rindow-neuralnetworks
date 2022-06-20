<?php
namespace Rindow\NeuralNetworks\Layer;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use Rindow\NeuralNetworks\Gradient\Core\Variable;
use Rindow\NeuralNetworks\Gradient\Core\Undetermined;
use Rindow\NeuralNetworks\Gradient\Core\UndeterminedNDArray;
use Rindow\NeuralNetworks\Gradient\Core\GradientTape;
use Rindow\NeuralNetworks\Gradient\Core\GradientUtils;
use Rindow\NeuralNetworks\Model\BuildContext;

/**
 *
 */
abstract class AbstractRNNLayer extends AbstractLayerBase implements RNNLayer
{
    use GradientUtils;
    abstract protected function call(NDArray $inputs, bool $training, array $initialStates=null, array $options=null);
    abstract protected function differentiate(NDArray $dOutputs, array $dStates=null);
    abstract protected function numOfOutputStates($options);

    protected $enableInitialStates;

    final public function forward(object $inputs, bool $training, array $initialStates=null,array $options=null)
    {
        if(BuildContext::$build) {
            $variables = null;
            if($inputs!==null) {
                $variables = [$inputs];
            }
            if($initialStates!==null) {
                $variables = array_merge($variables,$initialStates);
            }
            $results = $this->build($variables,$options);
            if(is_array($results)) {
                $outputs = array_shift($results);
                return [$outputs,$results];
            } else {
                return $outputs;
            }
        }
        $this->assertInputShape($inputs,'forward');
        $this->assertStatesShape($initialStates,'forward');
        $results = $this->call($inputs,$training,$initialStates,$options);
        if(is_array($results)) {
            [$outputs,$states] = $results;
            $this->assertStatesShape($states,'forward');
        } elseif($results instanceof NDArray) {
            $outputs = $results;
        }
        $this->assertOutputShape($outputs,'forward');
        return $results;
    }

    /**
    *  @param  array<NDArray> $dOutputs
    *  @return array<NDArray>
    */
    final public function backward(array $dOutputs) : array
    {
        $dStates = $dOutputs;
        $dOutputs = array_shift($dStates);
        if(!($dOutputs instanceof NDArray)) {
            throw new InvalidArgumentException('dOutputs must be list of NDArray');
        } elseif(count($dStates)==0) {
            $dStates = null;
        }

        $this->assertOutputShape($dOutputs,'backward');
        $this->assertStatesShape($dStates,'backward');

        $results = $this->differentiate($dOutputs,$dStates);

        if(is_array($results)) {
            [$dInputs,$dStates] = $results;
            $this->assertStatesShape($dStates,'backward');
            $results = array_merge([$dInputs],$dStates);
        } elseif($results instanceof NDArray) {
            $dInputs = $results;
            $results = [$results];
        }
        $this->assertInputShape($dInputs,'backward');
        return $results;
    }


    protected function callCell(NDArray $inputs,bool $training, array $initialStates=null, array $options=null)
    {
        $K = $this->backend;
        $this->enableInitialStates=($initialStates!==null)?true:false;
        [$batches,$timesteps,$feature]=$inputs->shape();
        if($initialStates===null&&
            $this->stateful) {
            $initialStates = $this->initialStates;
        }
        if($initialStates===null){
            $initialStates = [];
            foreach($this->statesShapes as $shape){
                $initialStates[] = $K->zeros(array_merge([$batches],$shape));
            }
        }
        $outputs = null;
        if($this->returnSequences){
            $outputs = $K->zeros([$batches,$timesteps,$this->units]);
        }
        [$outputs,$states,$calcStates] = $K->rnn(
            [$this->cell,'forward'],
            $inputs,
            $initialStates,
            $training,
            $outputs,
            $this->goBackwards
        );
        $this->calcStates = $calcStates;
        $this->origInputsShape = $inputs->shape();
        if($this->stateful) {
            $this->initialStates = $states;
        }
        if($this->returnState){
            return [$outputs,$states];
        } else {
            return $outputs;
        }
    }

    protected function differentiateCell(NDArray $dOutputs, array $dNextStates=null)
    {
        $K = $this->backend;
        $dInputs=$K->zeros($this->origInputsShape);
        if($dNextStates===null){
            $dNextStates = [];
            $batches = $dOutputs->shape()[0];
            foreach($this->statesShapes as $shape){
                $dNextStates[] = $K->zeros(array_merge([$batches],$shape),$dOutputs->dtype());
            }
        }

        $grads = $this->cell->getGrads();
        foreach($grads as $grad){
            $K->clear($grad);
        }
        [$dInputs,$dPrevStates] = $K->rnnBackward(
            [$this->cell,'backward'],
            $dOutputs,
            $dNextStates,
            $this->calcStates,
            $dInputs,
            $this->goBackwards
        );
        $this->calcStates = null;
        if($this->enableInitialStates) {
            return [$dInputs, $dPrevStates];
        } else {
            return $dInputs;
        }
    }

    /**
    *  @param Variable  $inputs
    *  @param bool      $training
    *  @param array<Variable> $initialStates
    *  @param array     $options
    *  @return array<Variable>
    *       outputs
    */
    public function __invoke($inputs, bool $training, array $initialStates=null, array $options=null)
    {
        $outputs = null;
        if($this->outputShape==null) {
            $inputShape = null;
            $creator = $inputs->creator();
            if($creator) {
                $inputShape = [$inputs];
            }
            $outputs = $this->build($inputShape);
        }
        if($inputs instanceof Undetermined) {
            if($outputs===null) {
                throw new InvalidArgumentException('Undetermined is found in second calling.');
            }
            if(is_array($outputs)) {
                $states = $outputs;
                $outputs = array_shift($states);
                return [$outputs,$states];
            } else {
                return $outputs;
            }
        }

        $inputsVariables = [$inputs];
        if($initialStates!==null) {
            $rawStatus = array_map(function($stat){return $stat->value();},$initialStates);
            $inputsVariables = array_merge($inputsVariables,$initialStates);
        } else {
            $rawStatus = null;
        }
        $outputs = $this->forward($inputs->value(),$training,$rawStatus,$options);
        if(is_array($outputs)) {
            [$o, $outputs] = $outputs;
            array_unshift($outputs, $o);
        } else {
            $outputs = [$outputs];
        }
        $outputsVariables = $this->postGradientProcess(
            $this->backend, $inputsVariables, $outputs);
        if(count($outputsVariables)>1) {
            $outputs = array_shift($outputsVariables);
            return [$outputs,$outputsVariables];
        } else {
            return $outputsVariables[0];
        }
    }
}
