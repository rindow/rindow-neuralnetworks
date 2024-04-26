<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use ArrayAccess;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\GradientTape;
use Rindow\NeuralNetworks\Gradient\Core\GradientUtils;
use Rindow\NeuralNetworks\Gradient\Variable;

/**
 *
 */
abstract class AbstractRNNLayer extends AbstractLayerBase implements RNNLayer
{
    use GradientUtils;
    //abstract protected function numOfOutputStates($options) : int;

    protected int $units;

    /** @var array<NDArray|null> $initialStates */
    protected array $initialStates; // the statefull variable is not in container
    /** @var array<array<int>> $statesShapes */
    protected array $statesShapes;
    //protected $calcStates;
    //protected $origInputsShape;
    //protected $enableInitialStates;
    protected string $kernelInitializerName;
    protected string $recurrentInitializerName;
    protected string $biasInitializerName;
    protected bool $returnSequences;
    protected bool $returnState;
    protected bool $goBackwards;
    protected bool $stateful;
    protected RNNCell $cell;

    protected function setUnits(int $units) : void
    {
        $this->units = $units;
    }

    protected function setKernelInitializerNames(
        mixed $kernelInitializerName=null,
        mixed $recurrentInitializerName=null,
        mixed $biasInitializerName=null,
    ) : void
    {
        $this->kernelInitializerName = $this->toStringName($kernelInitializerName);
        $this->recurrentInitializerName = $this->toStringName($recurrentInitializerName);
        $this->biasInitializerName = $this->toStringName($biasInitializerName);
    }

    protected function setFlags(
        bool $returnSequences=null,
        bool $returnState=null,
        bool $goBackwards=null,
        bool $stateful=null,
        ) : void
    {
        $this->returnSequences = $returnSequences;
        $this->returnState = $returnState;
        $this->goBackwards = $goBackwards;
        $this->stateful = $stateful;
    }

    protected function setCell(RNNCell $cell) : void
    {
        $this->cell = $cell;
    }

    protected function cell() : RNNCell
    {
        return $this->cell;
    }

    /**
     * @param array<NDArray> $states
     */
    protected function assertStatesShape(array $states=null,string $direction) : void
    {
        if(!$this->shapeInspection)
            return;
        if($states===null) {
            return;
        }
        //if($this->statesShapes===null) {
        //    throw new InvalidArgumentException('Uninitialized status shape');
        //}
        if(count($states)!=count($this->statesShapes)){
            throw new InvalidArgumentException('Unmatch num of status. status need '.count($this->statesShapes).' NDArray. '.count($states).'given.');
        }
        foreach($states as $idx=>$state){;
            $stateShape = $this->statesShapes[$idx];
            $shape = $state->shape();
            $batchNum = array_shift($shape);
            if($shape!=$stateShape) {
                $shape = $this->shapeToString($shape);
                $stateShape = $this->shapeToString($stateShape);
                $name = $this->name ?? $this->basename($this);
                throw new InvalidArgumentException('Shape of state'.$idx.' must be '.$stateShape.', '.$shape.' given in '.$name.':'.$direction);
            }
        }
    }

    /**
     * @return array<array<int>>
     */
    public function statesShapes() : array
    {
        return $this->statesShapes;
    }

    public function setShapeInspection(bool $enable) : void
    {
        parent::setShapeInspection($enable);
        $this->cell->setShapeInspection($enable);
    }

    public function getParams() : array
    {
        return $this->cell->getParams();
    }

    public function getGrads() : array
    {
        return $this->cell->getGrads();
    }

    public function reverseSyncWeightVariables() : void
    {
        $this->cell->reverseSyncCellWeightVariables($this->weights);
    }

    /*
     *  param  array<NDArray> $dOutputs
     *  return array<NDArray>
     */
    final public function backward(
        array $dOutputs,
        ArrayAccess $grads=null,
        array $oidsToCollect=null
        ) : array
    {
        if(!$this->shapeInspection) {
            $tmpdStates = $dOutputs;
            $tmpdOutputs = array_shift($tmpdStates);
            if(!($tmpdOutputs instanceof NDArray)) {
                throw new InvalidArgumentException('dOutputs must be list of NDArray');
            } elseif(count($tmpdStates)==0) {
                $tmpdStates = null;
            }
            $this->assertOutputShape($tmpdOutputs,'backward');
            $this->assertStatesShape($tmpdStates,'backward');
        }

        $dInputs = $this->differentiate($dOutputs);
        if(!$this->shapeInspection) {
            $tmpdStates = $dInputs;
            $tmpdInputs = array_shift($tmpdStates);
            if(count($tmpdStates)>0) {
                $this->assertStatesShape($tmpdStates,'backward');
            }
            $this->assertInputShape($tmpdInputs,'backward');
        }
        $this->collectGradients($this->backend,array_map(null,$this->trainableVariables(),$this->getGrads()),
            $grads,$oidsToCollect);

        return $dInputs;
    }

    /**
     * @param array<NDArray> $inputs
     * @return array<NDArray>
     */
    protected function call(array $inputs,bool $training=null) : array
    {
        $K = $this->backend;
        $container = $this->container();
        $initialStates = $inputs;
        $inputs = array_shift($initialStates);
        $container->enableInitialStates=(count($initialStates)>0)?true:false;
        [$batches,$timesteps,$feature]=$inputs->shape();
        if(count($initialStates)==0 && $this->stateful) {
            $initialStates = $this->initialStates; // the statefull variable is not in container
        }
        if(count($initialStates)==0){
            foreach($this->statesShapes as $shape){
                $initialStates[] = $K->zeros(array_merge([$batches],$shape));
            }
        } else {
            $states = [];
            foreach($initialStates as $i => $s) {
                if($s===null) {
                    $shape = $this->statesShapes[$i];
                    $states[] = $K->zeros(array_merge([$batches],$shape));
                } else {
                    $states[] = $s;
                }
            }
            $initialStates = $states;
            unset($states);
        }
        
        $outputs = null;
        if($this->returnSequences){
            $outputs = $K->zeros([$batches,$timesteps,$this->units]);
        }
        [$outputs,$states,$calcStates] = $K->rnn(
            [$this->cell,'forward'],
            $inputs,
            $initialStates,
            training:$training,
            outputs:$outputs,
            goBackwards:$this->goBackwards
        );
        $container->calcStates = $calcStates;
        $container->origInputsShape = $inputs->shape();
        if($this->stateful) {
            $this->initialStates = $states; // the statefull variable is not in container
        }
        if($this->returnState){
            return array_merge([$outputs],$states);
        } else {
            return [$outputs];
        }
    }

    /**
     * @param array<NDArray> $dOutputs
     * @return array<NDArray>
     */
    protected function differentiate(array $dOutputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        $dNextStates = $dOutputs;
        $dOutputs = array_shift($dNextStates);

        $dInputs=$K->zeros($container->origInputsShape);
        if(count($dNextStates)==0){
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
            $container->calcStates,
            $dInputs,
            $this->goBackwards
        );
        $container->calcStates = null;
        if($container->enableInitialStates) {
            return array_merge([$dInputs], $dPrevStates);
        } else {
            return [$dInputs];
        }
    }

    /**
     * @return NDArray|array<Variable>
     */
    public function __invoke(mixed ...$args) : NDArray|array
    {
        return $this->forward(...$args);
    }

    /*
     * param Variable  $inputs
     * param bool      $training
     * param array<Variable> $initialStates
     * return NDArray|array<Variable> outputs
     */
    final public function forward(
        object $inputs,
        Variable|bool $training=null,
        array $initialStates=null
        ) : NDArray|array
    {
        $inputs = [$inputs];
        if($initialStates!==null) {
            $inputs = array_merge($inputs,$initialStates);
        }
        [$inputs,$rawInputs]     = $this->packAndUnpackVariables($this->backend,$inputs);
        if($training===null) {
            $rawTraining = null;
            $options = null;
        } else {
            [$training,$rawTraining] = $this->packAndUnpackVariable($this->backend,$training);
            $options = ['training'=>$training];
        }

        if(!$this->built) {
            $this->build($inputs);
            $this->built = true;
        }

        $session = $this->preGradientProcessOnSession($inputs,$options);
        $session->begin();
        try {
            $rawInitialStates = $rawInputs;
            $tmpRawInputs = array_shift($rawInitialStates);
            $this->assertInputShape($tmpRawInputs,'forward');
            if(count($rawInitialStates)>0) {
                $this->assertStatesShape($rawInitialStates,'forward');
            }
            unset($tmpRawInputs);
            unset($rawInitialStates);
            $rawOutputs = $this->call($rawInputs,training:$rawTraining);
            $rawStates = $rawOutputs;
            $tmpRawOutputs = array_shift($rawStates);
            if(count($rawStates)>0) {
                $this->assertStatesShape($rawStates,'forward');
            }
            $this->assertOutputShape($tmpRawOutputs,'forward');
            unset($tmpRawOutputs);
            unset($rawStates);
        } finally {
            $session->end();
        }

        $outputs = $this->postGradientProcessOnSession(
            $this->backend, $session, $inputs, $rawOutputs);
        
        if(count($outputs)>1) {
            $states = $outputs;
            $outputs = array_shift($states);
            return [$outputs,$states];
        } else {
            return $outputs[0];
        }
    }

    /**
     * Call from SessionFunc in compiled graph
     * @param array<NDArray> $inputs
     * @param array<string,mixed> $options
     * @return array<NDArray>
     */
    public function _rawCall(array $inputs,array $options) : array
    {
        $training = $options['training'] ?? false;
        $results = $this->call($inputs,training:$training);
        return $results;
    }

    public function __clone()
    {
        if(isset($this->cell)) {
            $this->cell = clone $this->cell;
        }
        $this->allocateWeights(count($this->weights));
        if($this->assignedWeights) {
            $this->syncWeightVariables();
        }
    }
}
