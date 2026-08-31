<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;

class Tanh extends AbstractFunction
{
    protected function call(array $inputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        $outputs = $K->tanh($inputs[0]);
        $container->outputs = $outputs;
        return [$outputs];
    }

    protected function differentiate(array $dOutputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        $outputs = $container->outputs;
        // dx = dy * (1 - y**2)
        $dInputs = $K->mul(
            $dOutputs[0],
            $K->increment(
                $K->square($outputs),
                beta:1,
                alpha:-1,
            )
        );
        return [$dInputs];
    }
}
