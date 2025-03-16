<?php
namespace Rindow\NeuralNetworks\Layer;

use ArrayAccess;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Variable;

/**
 *
 */
interface RNNLayer extends Layer
{
    /**
     * @param array<Variable> $initialStates
     * @return Variable|array{Variable,array<Variable>}
     */
    public function forward(
        object $inputs,
        ?bool $training=null,
        ?array $initialStates=null,
        ?NDArray $mask=null,
    ) : Variable|array;

    /**
     * @param array<NDArray> $dOutputs
     * @param ArrayAccess<object,object> $grads
     * @param array<NDArray> $oidsToCollect
     * @return array<NDArray>
     */
    public function backward(
        array $dOutputs,
        ?ArrayAccess $grads=null,
        ?array $oidsToCollect=null
    ) : array;
}
