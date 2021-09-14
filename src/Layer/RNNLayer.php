<?php
namespace Rindow\NeuralNetworks\Layer;

use Interop\Polite\Math\Matrix\NDArray;

/**
 *
 */
interface RNNLayer extends LayerBase
{
    public function forward(object $inputs, bool $training,array $initialStates=null,array $options=null);
    public function backward(array $dOutputs) : array; // dOutputs is NDArray or array
}
