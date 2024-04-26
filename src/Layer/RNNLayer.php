<?php
namespace Rindow\NeuralNetworks\Layer;

use ArrayAccess;
use Interop\Polite\Math\Matrix\NDArray;

/**
 *
 */
interface RNNLayer extends Layer
{
    /**
     * @param array<NDArray> $initialStates
     * @return NDArray|array<NDArray>
     */
    public function forward(
        object $inputs,
        bool $training=null,
        array $initialStates=null
    ) : NDArray|array;

    /**
     * @param array<NDArray> $dOutputs
     * @param ArrayAccess<object,object> $grads
     * @param array<NDArray> $oidsToCollect
     * @return array<NDArray>
     */
    public function backward(
        array $dOutputs,
        ArrayAccess $grads=null,
        array $oidsToCollect=null
    ) : array;
}
