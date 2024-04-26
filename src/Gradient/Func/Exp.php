<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;

class Exp extends AbstractFunction
{
    protected function call(array $inputs) : array
    {
        $container = $this->container();
        $outputs = $this->backend->exp($inputs[0]);
        $container->outputs = $outputs;
        return [$outputs];
    }

    protected function differentiate(array $dOutputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        $dInput = $K->mul($container->outputs,$dOutputs[0]);
        $container->outputs = null;
        return [$dInput];
    }
}
