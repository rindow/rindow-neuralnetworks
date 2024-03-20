<?php
namespace Rindow\NeuralNetworks\Layer;

use Interop\Polite\Math\Matrix\NDArray;

/**
 *
 */
interface RNNCell extends Layer
{
    public function forward(NDArray $inputs, array $states, bool $training=null, object $calcState=null) : array;
    public function backward(NDArray $dOutputs, array $dStates, object $calcState) : array;
}
