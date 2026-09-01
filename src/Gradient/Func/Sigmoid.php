<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;

class Sigmoid extends AbstractFunction
{
    protected function call(array $inputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        $outputs = $K->sigmoid($inputs[0]);
        $container->inputs = $inputs[0];
        $container->outputs = $outputs;
        return [$outputs];
    }

    protected function differentiate(array $dOutputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        $outputs = $container->outputs;
        $dInput = $K->dSigmoid($dOutputs[0], $outputs);
        $container->inputs = null;
        $container->outputs = null;
        return [$dInput];
    }
}
