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
    /**
     * @param array<NDArray> $dOutputs
     * @param ArrayAccess<object,object> $grads
     * @param array<object> $oidsToCollect
     * @return array<NDArray>
     */
    public function backward(array $dOutputs, ArrayAccess $grads=null, array $oidsToCollect=null) : array;
}
