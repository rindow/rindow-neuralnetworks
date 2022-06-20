<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;

class Log extends AbstractFunction
{
    protected function call(array $inputs) : array
    {
        return [$this->backend->log($inputs[0])];
    }

    protected function differentiate(array $dOutputs) : array
    {
        $K = $this->backend;
        $inputs = $this->inputsVariables;
        $x = $inputs[0]->value();
        $dInput = $K->div($dOutputs[0],$x);
        return [$dInput];
    }
}
