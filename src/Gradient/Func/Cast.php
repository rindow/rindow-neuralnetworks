<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;
use LogicException;

class Cast extends AbstractFunction
{
    protected int $dtype;

    public function __construct(
        object $backend,
        int $dtype,
        ?string $name=null,
    )
    {
        parent::__construct($backend,name:$name);
        $this->dtype = $dtype;
    }

    /**
    *  @param array<NDArray>  $inputs
    *       inputs
    *  @return array<NDArray>
    *       outputs
    */
    protected function call(array $inputs) : array
    {
        $output = $this->backend->cast($inputs[0],$this->dtype);
        $this->unbackpropagatables = [true];
        return [$output];
    }

    /**
    *  @param array<NDArray>  $dOutputs
    *  @return array<NDArray>
    */
    protected function differentiate(array $dOutputs) : array
    {
        throw new LogicException('This function is not differentiable');
    }
}
