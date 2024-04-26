<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use InvalidArgumentException;
use LogicException;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;
use Rindow\NeuralNetworks\Gradient\Core\ArrayShape;
use Rindow\NeuralNetworks\Gradient\ArrayShape as ArrayShapeInterface;
use Interop\Polite\Math\Matrix\NDArray;

class Shape extends AbstractFunction
{
    /**
     * @param array<NDArray> $inputs
     * @return array<ArrayShapeInterface>
     */
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
