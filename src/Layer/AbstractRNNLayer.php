<?php
namespace Rindow\NeuralNetworks\Layer;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

/**
 *
 */
abstract class AbstractRNNLayer extends AbstractLayerBase implements RNNLayer
{
    abstract protected function call(NDArray $inputs, bool $training, array $initialStates=null, array $options=null);
    abstract protected function differentiate(NDArray $dOutputs, array $dStates=null);

    final public function forward(NDArray $inputs, bool $training, array $initialStates=null,array $options=null)
    {
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

    final public function backward(NDArray $dOutputs, array $dStates=null)
    {
        $this->assertOutputShape($dOutputs,'backward');
        $this->assertStatesShape($dStates,'backward');

        $results = $this->differentiate($dOutputs,$dStates);

        if(is_array($results)) {
            [$dInputs,$dStates] = $results;
            $this->assertStatesShape($dStates,'backward');
        } elseif($results instanceof NDArray) {
            $dInputs = $results;
        }
        $this->assertInputShape($dInputs,'backward');
        return $results;
    }


    protected function callCell(NDArray $inputs,bool $training, array $initialStates=null, array $options=null)
    {
        $K = $this->backend;
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
        if($this->returnState) {
            return [$dInputs, $dPrevStates];
        } else {
            return $dInputs;
        }
    }
}
