<?php
namespace Rindow\NeuralNetworks\Layer;

use ArrayAccess;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Variable;

/**
 *
 */
interface SequentialLayer extends Layer
{
    public function forward(NDArray $inputs, Variable|bool $training=null) : Variable;
    public function backward(array $dOutputs, ArrayAccess $grads=null, array $oidsToCollect=null) : array;
}
