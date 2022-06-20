<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;

class Square extends AbstractFunction
{
    protected function call(array $inputs) : array
    {
        return [$this->backend->square($inputs[0])];
    }

    protected function differentiate(array $dOutputs) : array
    {
        $K = $this->backend;
        $inputs = $this->inputsVariables;
        $x = $inputs[0]->value();
        $dInput = $K->scale(2,$K->mul($x,$dOutputs[0]));
        return [$dInput];
    }
}
