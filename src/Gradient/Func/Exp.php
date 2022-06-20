<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;

class Exp extends AbstractFunction
{
    protected $outputs;

    protected function call(array $inputs) : array
    {
        $outputs = $this->backend->exp($inputs[0]);
        $this->outputs = $outputs;
        return [$outputs];
    }

    protected function differentiate(array $dOutputs) : array
    {
        $K = $this->backend;
        $dInput = $K->mul($this->outputs,$dOutputs[0]);
        $this->outputs = null;
        return [$dInput];
    }
}
