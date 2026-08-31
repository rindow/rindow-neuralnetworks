<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;

class Log1p extends AbstractFunction
{
    protected function call(array $inputs) : array
    {
        $container = $this->container();
        $container->inputs = $inputs;
        return [$this->backend->log1p($inputs[0])];
    }

    protected function differentiate(array $dOutputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        $x = $container->inputs[0];
        $dInput = $K->dLog1p($dOutputs[0],$x);
        return [$dInput];
    }
}
