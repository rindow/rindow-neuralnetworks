<?php
namespace Rindow\NeuralNetworks\Metric;

use Interop\Polite\Math\Matrix\NDArray;

/**
 *
 */
interface Metric
{
    public function name() : string;
    public function reset() : void;
    public function result() : float;
    public function immediateUpdate(float $value) : void;
    public function update(NDArray $trues, NDArray $predicts) : void;
    public function forward(NDArray $trues, NDArray $predicts) : float;
    public function __invoke(mixed ...$args) : mixed;
}
