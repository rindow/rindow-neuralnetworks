<?php
namespace Rindow\NeuralNetworks\Layer;

use Interop\Polite\Math\Matrix\NDArray;

/**
 *
 */
interface RNNCell extends LayerBase
{
    public function forward(NDArray $inputs, array $states, bool $training, object $calcState, array $options=null) : array;
    public function backward(NDArray $dOutputs, array $dStates, object $calcState) : array;
}
