<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;

class Relu extends AbstractFunction
{
    protected function call(array $inputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        $outputs = $K->relu($inputs[0]);
        $container->inputs = $inputs[0];
        return [$outputs];
    }

    protected function differentiate(array $dOutputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        $inputs = $container->inputs;
        $mask = $K->greater($inputs,0.0);
        $dInputs = $K->mul($dOutputs[0],$mask);
        $container->inputs = null;
        return [$dInputs];
    }
}
