<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use ArrayAccess;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Gradient\Core\GradientTape;
use Rindow\NeuralNetworks\Gradient\Core\GradientUtils;
use Rindow\NeuralNetworks\Gradient\MaskedNDArray;

/**
 *
 */
abstract class AbstractLayer extends AbstractLayerBase implements SequentialLayer
{
    use GradientUtils;
    abstract protected function call(NDArray $inputs, ?bool $training=null) : NDArray;
    abstract protected function differentiate(NDArray $dOutputs) : NDArray;

    /**
    *  @param  array<NDArray> $dOutputs
    *  @return array<NDArray>
    */
    final public function backward(array $dOutputs, ?ArrayAccess $grads=null, ?array $oidsToCollect=null) : array
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
        $this->assertInputShape($dInputs,'backward');
        $this->collectGradients($this->backend,array_map(null,$this->trainableVariables(),$this->getGrads()),
            $grads,$oidsToCollect);
        return [$dInputs];
    }


    final public function __invoke(mixed ...$args) : mixed
    {
        return $this->forward(...$args);
    }
    
    public function forward(NDArray $inputs, Variable|bool|null $training=null) : Variable
    {
        [$inputs,$rawInputs]     = $this->packAndUnpackVariable($this->backend,$inputs);
        //[$training,$rawTraining] = $this->packAndUnpackVariable($this->backend,$training);
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

        $session = $this->preGradientProcessOnSession([$inputs],$options);
        $session->begin();
        try {
            $this->assertInputShape($rawInputs,'forward');
            $rawOutputs = $this->call($rawInputs, training:$rawTraining);
            $rawOutputs = $this->makeSingleMaskedValue($rawInputs, $rawOutputs);
            $this->assertOutputShape($rawOutputs,'forward');
        } finally {
            $session->end();
        }

        $outputs = $this->postGradientProcessOnSession(
            $this->backend, $session, [$inputs], [$rawOutputs]
        );
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
        //$training = $options['training'] ?? null;
        //$outputs = $this->call($inputs[0],training:$training);
        $input = $inputs[0];
        $output = $this->call($input, ...$options);
        $output = $this->makeSingleMaskedValue($input, $output);
        return [$output];
    }

    public function computeMask(
        array|NDArray $inputs,
        array|NDArray|null $previousMask
        ) : array|NDArray|null
    {
        return $previousMask;
    }

    /**
     * @param array<NDArray> $dOutputs
     * @return array<NDArray>
     */
    public function _rawDifferentiate(array $dOutputs) : array
    {
        $results = $this->differentiate($dOutputs[0]);
        return [$results];
    }

    public function __clone()
    {
        if(isset($this->kernel)) {
            $this->kernel = clone $this->kernel;
        }
        if(isset($this->bias)) {
            $this->bias = clone $this->bias;
        }
        if(isset($this->dKernel)) {
            $this->dKernel = clone $this->dKernel;
        }
        if(isset($this->dBias)) {
            $this->dBias = clone $this->dBias;
        }
        $this->allocateWeights(array_map(fn($weight)=>$weight->name(),$this->weights));
        if($this->assignedWeights) {
            $this->syncWeightVariables();
        }
    }
}
