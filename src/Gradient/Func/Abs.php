<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;

class Abs extends AbstractFunction
{
    protected function call(array $inputs) : array
    {
        $container = $this->container();
        $outputs = $this->backend->abs($inputs[0]);
        $container->input = $inputs[0];
        return [$outputs];
    }

    protected function differentiate(array $dOutputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        $input = $container->input;
        $dInputs = $K->mul($K->sign($input),$dOutputs[0]);
        $container->input = null;
        return [$dInputs];
    }
}
