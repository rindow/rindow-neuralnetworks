<?php
namespace Rindow\NeuralNetworks\Layer;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\Variable;

/**
 *
 */
interface RNNCell extends Layer
{
    /**
     * @param array<NDArray> $states
     * @return array<NDArray>
     */
    public function forward(
        NDArray $inputs,
        array $states,
        ?bool $training=null,
        ?object $calcState=null
    ) : array;

    /**
     * @param array<NDArray> $dStates
     * @return array{NDArray,array<NDArray>}
     */
    public function backward(
        array $dStates,
        object $calcState
    ) : array;

    /**
     * @param array<Variable> $weights
     */
    public function reverseSyncCellWeightVariables(array $weights) : void;
}
