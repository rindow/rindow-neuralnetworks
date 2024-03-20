<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;

class Bandpart extends AbstractFunction
{
    protected int $lower;
    protected int $upper;

    public function __construct(
        object $backend,
        int $lower,
        int $upper,
    )
    {
        parent::__construct($backend);
        $this->lower = $lower;
        $this->upper = $upper;
    }

    protected function call(array $inputs) : array
    {
        $K = $this->backend;
        $output = $K->bandpart($inputs[0],$this->lower,$this->upper);
        $this->unbackpropagatables = [true];
        return [$output];
    }

    protected function differentiate(array $dOutputs) : array
    {
        throw new LogicException('This function is not differentiable');
    }
}
