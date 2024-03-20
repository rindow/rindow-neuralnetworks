<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use InvalidArgumentException;
use Rindow\NeuralNetworks\Gradient\ArrayShape as ArrayShapeInterface;
use Rindow\NeuralNetworks\Gradient\Scalar as ScalarInterface;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;
use Rindow\NeuralNetworks\Gradient\Core\Scalar;
use Interop\Polite\Math\Matrix\NDArray;

class Ones extends Zeros
{
    /**
    *  @param array<NDArray>  $inputs
    *       inputs
    *  @return array<NDArray>
    *       outputs
    */
    protected function call(array $inputs) : array
    {
        $K = $this->backend;
        $shape = $this->translateToShape($inputs);
        $this->unbackpropagatables = [true];
        return [$K->ones($shape,dtype:$this->dtype)];
    }
}
