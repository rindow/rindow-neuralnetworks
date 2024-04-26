<?php
namespace Rindow\NeuralNetworks\Optimizer;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Variable;

/**
 *
 */
interface Optimizer
{
    /**
     * @param array<NDArray|Variable> $params
     */
    public function build(array $params) : void;
    
    /**
     * @param array<NDArray|Variable> $params   weight paramators
     * @param array<NDArray|Variable> $grads    gradients
     */
    public function update(array $params, array $grads) : void;

    /**
     * @return array<NDArray>
     */
    public function getWeights() : array;

    /**
     * @param array<NDArray> $params
     */
    public function loadWeights(array $params) : void;

    /**
     * @return array<string,mixed>
     */
    public function getConfig() : array;
}
