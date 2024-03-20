<?php
namespace Rindow\NeuralNetworks\Gradient\Core;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;

class StopGradient extends AbstractFunction
{
    protected function call(array $inputs) : array
    {
        $K = $this->backend;
        $outputs = [$K->copy($inputs[0])];
        $this->unbackpropagatables = [true];
        return $outputs;
    }

    protected function differentiate(array $dOutputs) : array
    {
        $K = $this->backend;
        $dInputs = [$K->zerosLike($dOutputs[0])];
        return $dInputs;
    }
}
