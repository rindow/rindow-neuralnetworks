<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use ArrayAccess;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\GradientUtils;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Gradient\MaskedNDArray;

abstract class AbstractMultiInputLayer extends AbstractLayerBase
{
    use GradientUtils;

    /**
     * @param array<NDArray> $inputs
     */
    abstract protected function call(array $inputs, ?bool $training=null) : NDArray;

    /**
     * @return array<NDArray>
     */
    abstract protected function differentiate(NDArray $dOutputs) : array;

    /**
     * @param array<NDArray> $inputs
     */
    protected function assertInputShapes(array $inputs, string $direction) : void
    {
        if(!$this->shapeInspection)
            return;
        if($this->inputShape===null) {
            throw new InvalidArgumentException('Uninitialized input shape');
        }
        if(count($inputs)!=count($this->inputShape)){
            throw new InvalidArgumentException('Unmatch num of input. inputs need '.count($this->inputShape).' NDArray. '.count($inputs).'given in '.$this->name.':'.$direction);
        }
        $batchSize = null;
        foreach($inputs as $idx=>$input){;
            $inputShape = $this->inputShape[$idx];
            $shape = $input->shape();
            if($batchSize === null) {
                $batchSize = array_shift($shape);
            } else {
                if($batchSize != array_shift($shape)) {
                    throw new InvalidArgumentException('unmatch batch size of input '.$idx.': ['.$batchSize.'] in '.$this->name.':'.$direction);
                }
            }
            if($shape!=$inputShape) {
                $shape = $shape ? implode(',',$shape) : '';
                throw new InvalidArgumentException('unmatch shape of input '.$idx.': ['.$shape.'], must be ['.implode(',',$inputShape).'] in '.$this->name.':'.$direction);
            }
        }
    }

    /**
     * @param array<NDArray> $dOutputs
     * @param ArrayAccess<object,object> $grads
     * @param array<object> $oidsToCollect
     * @return array<NDArray>
     */
    public function backward(array $dOutputs, ?ArrayAccess $grads=null,?array $oidsToCollect=null) : array
    {
        if(count($dOutputs)!=1) {
            throw new InvalidArgumentException('dOutputs must be list containing one NDArray');
        }
        $dOutputs = $dOutputs[0];
        if(!($dOutputs instanceof NDArray)) {
            throw new InvalidArgumentException('dOutputs must be list containing one NDArray');
        }
        $this->assertOutputShape($dOutputs,'backward');
        $dInputs = $this->differentiate($dOutputs);
        $this->assertInputShapes($dInputs,'backward');
        $this->collectGradients($this->backend,array_map(null,$this->trainableVariables(),$this->getGrads()),
            $grads,$oidsToCollect);
        return $dInputs;
    }

    /**
    *  @param array<Variable>  $inputs
    *  @return Variable
    */
    public function __invoke(array $inputs, Variable|bool|null $training=null) : Variable
    {
        $outputs = $this->forward($inputs, $training);
        return $outputs;
    }

    /**
    *  @param array<Variable>  $inputs
    *  @return Variable
    */
    public function forward(array $inputs, Variable|bool|null $training=null) : Variable
    {
        if(count($inputs)<2) {
            throw new InvalidArgumentException('Must have arguments greater than 2 or equal');
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
        $session = $this->preGradientProcessOnSession($inputs);
        $session->begin();
        try {
            $this->assertInputShapes($inputs,'forward');
            $rawOutputs = $this->call($rawInputs,$rawTraining);
            $rawOutputs = $this->makeMultiMaskedValues($rawInputs, [$rawOutputs]);
            $this->assertOutputShape($rawOutputs[0],'forward');
        } finally {
            $session->end();
        }
        $outputs = $this->postGradientProcessOnSession(
            $this->backend, $session, $inputs, $rawOutputs);
        return $outputs[0];
    }

    /**
     * Call from SessionFunc in compiled graph
     * @param array<NDArray> $inputs
     * @param array<string,mixed> $options
     * @return array<NDArray>
     */
    public function _rawCall(array $inputs,array $options) : array
    {
        $training = $options['training'] ?? null;
        $outputs = $this->call($inputs, training:$training);

        $values = $this->makeMultiMaskedValues($inputs, [$outputs]);
        return $values;
    }

    public function computeMask(
        array|NDArray $inputs,
        array|NDArray|null $previousMask
        ) : array|NDArray|null
    {
        return null;
    }
}
