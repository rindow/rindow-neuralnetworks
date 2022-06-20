<?php
namespace Rindow\NeuralNetworks\Optimizer;

/**
 *
 */
interface Optimizer
{
    /**
     * @param array<NDArray> $params  weight paramators
     * @param array<NDArray> $grads   gradients
     */
    public function build(array $params) : void;
    public function update(array $params, array $grads) : void;
    public function getWeights() : array;
    public function getConfig() : array;
}
