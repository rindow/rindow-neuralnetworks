<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;
use LogicException;

class NotEqual extends AbstractFunction
{
    protected $numOfInputs = 2;

    /**
    *  @param array<NDArray>  $inputs
    *       inputs
    *  @return array<NDArray>
    *       outputs
    */
    protected function call(array $inputs) : array
    {
        $container = $this->container();
        $container->inputs = $inputs;

        $output = $this->backend->notEqual($inputs[0],$inputs[1]);
        $this->unbackpropagatables = [true];
        return [$output];
    }

    /**
    *  @param array<NDArray>  $dOutputs
    *       difference outputs
    *  @return array<NDArray>
    *       difference inputs
    */
    protected function differentiate(array $dOutputs) : array
    {
        throw new LogicException('This function is not differentiable');
    }
}
