<?php
namespace Rindow\NeuralNetworks\Gradient\Core;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;

class StopGradient extends AbstractFunction
{
    protected function call(array $inputs) : array
    {
        return [$this->backend->copy($inputs[0])];
    }

    protected function differentiate(array $dOutputs) : array
    {
        $dInput = $this->backend->zerosLike($dOutputs[0]);
        return [$dInput];
    }
}
