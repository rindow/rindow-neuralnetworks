<?php
namespace Rindow\NeuralNetworks\Layer;

use ArrayAccess;
use Interop\Polite\Math\Matrix\NDArray;

/**
 *
 */
interface RNNLayer extends Layer
{
    public function forward(object $inputs, bool $training,array $initialStates=null,array $options=null);
    public function backward(array $dOutputs,ArrayAccess $grads=null,array $oidsToCollect=null) : array; // dOutputs is NDArray or array
}
