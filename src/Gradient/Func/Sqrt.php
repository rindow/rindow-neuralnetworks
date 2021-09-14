<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;

class Sqrt extends AbstractFunction
{
    protected function call(array $inputs) : array
    {
        return [$this->backend->sqrt($inputs[0])];
    }

    protected function differentiate(array $dOutputs) : array
    {
        $K = $this->backend;
        $inputs = $this->inputsVariables;
        $x = $inputs[0]->value();
        $dInput = $K->mul($dOutputs[0],$K->rsqrt($x,null,2.0));
        return [$dInput];
    }
}
