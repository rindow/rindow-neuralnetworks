<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use InvalidArgumentException;
use LogicException;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;
use Rindow\NeuralNetworks\Gradient\Core\ArrayShape;

class Shape extends AbstractFunction
{
    protected $outputs;

    protected function call(array $inputs) : array
    {
        $shape = $inputs[0]->shape();
        $outputs = new ArrayShape($shape);
        $this->unbackpropagatables = [true];
        return [$outputs];
    }

    protected function differentiate(array $dOutputs) : array
    {
        throw new LogicException('This function is not differentiable');
    }
}
