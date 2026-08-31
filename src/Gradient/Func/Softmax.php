<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;

class Softmax extends AbstractFunction
{
    protected function call(array $inputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        $outputs = $K->softmax($inputs[0]);
        $container->outputs = $outputs;
        return [$outputs];
    }

    protected function differentiate(array $dOutputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        $dInput = $K->dSoftmax($dOutputs[0], $container->outputs);
        return [$dInput];
    }
}
