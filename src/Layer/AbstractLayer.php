<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use LogicException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\Variable;
use Rindow\NeuralNetworks\Gradient\Core\Undetermined;
use Rindow\NeuralNetworks\Gradient\Core\UndeterminedNDArray;
use Rindow\NeuralNetworks\Gradient\Core\GradientTape;
use Rindow\NeuralNetworks\Gradient\Core\GradientUtils;
use Rindow\NeuralNetworks\Model\BuildContext;

/**
 *
 */
abstract class AbstractLayer extends AbstractLayerBase
{
    use GradientUtils;
    abstract protected function call(NDArray $inputs, bool $training) : NDArray;
    abstract protected function differentiate(NDArray $dOutputs) : NDArray;

    final public function forward(object $inputs, bool $training)
    {
        if(BuildContext::$build) {
            return $this->build($inputs);
        }
        $this->assertInputShape($inputs,'forward');

        $outputs = $this->call($inputs, $training);

        $this->assertOutputShape($outputs,'forward');
        return $outputs;
    }

    /**
    *  @param  array<NDArray> $dOutputs
    *  @return array<NDArray>
    */
    final public function backward(array $dOutputs) : array
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
        return [$dInputs];
    }

    /**
    *  @param Variable  $inputs
    *       inputs
    *  @return array<Variable>
    *       outputs
    */
    public function __invoke($inputs, bool $training)
    {
        $outputs = null;
        if($this->outputShape==null) {
            $inputShape = null;
            $creator = $inputs->creator();
            if($creator) {
                $inputShape = $inputs;
            }
            $outputs = $this->build($inputShape);
        }
        if($inputs instanceof Undetermined) {
            if($outputs===null) {
                throw new InvalidArgumentException('Undetermined is found in second calling.');
            }
            return $outputs;
        }
        $outputs = $this->forward($inputs->value(),$training);
        $outputs = $this->postGradientProcess(
            $this->backend, [$inputs], [$outputs]);
        return $outputs[0];
    }
}
