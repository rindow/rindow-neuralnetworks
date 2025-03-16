<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use InvalidArgumentException;
use LogicException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;

class OnesLike extends AbstractFunction
{
    protected function call(array $inputs) : array
    {
        $this->unbackpropagatables = [true];
        return [$this->backend->onesLike($inputs[0])];
    }

    protected function differentiate(array $dOutputs) : array
    {
        throw new LogicException('This function is not differentiable');
    }
}
