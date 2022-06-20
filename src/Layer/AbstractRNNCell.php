<?php
namespace Rindow\NeuralNetworks\Layer;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

/**
 *
 */
abstract class AbstractRNNCell extends AbstractLayerBase implements RNNCell
{
    abstract protected function call(NDArray $inputs, array $states, bool $training, object $calcState, array $options=null) : array;
    abstract protected function differentiate(NDArray $dOutputs, array $dStates, object $calcState) : array;

    final public function forward(NDArray $inputs, array $states, bool $training, object $calcState, array $options=null) : array
    {
        $this->assertInputShape($inputs,'forward');

        [$outputs,$states] = $this->call($inputs,$states,$training,$calcState,$options);

        $this->assertOutputShape($outputs,'forward');
        return [$outputs,$states];
    }

    final public function backward(NDArray $dOutputs, array $dStates, object $calcState) : array
    {
        $this->assertOutputShape($dOutputs,'backward');

        [$dInputs,$dStates] = $this->differentiate($dOutputs,$dStates,$calcState);

        $this->assertInputShape($dInputs,'backward');
        return [$dInputs,$dStates];
    }
}
