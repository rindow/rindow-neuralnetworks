<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;
use Rindow\NeuralNetworks\Gradient\Core\NullValue;

class ExpandDims extends AbstractFunction
{
    protected int $axis;

    public function __construct(
        object $backend,
        int $axis,
        ?string $name=null,
    )
    {
        parent::__construct($backend,name:$name);
        $this->axis = $axis;
    }

    protected function call(array $inputs) : array
    {
        $K = $this->backend;
        $outputs = $K->expandDims(
            $inputs[0],
            $this->axis,
        );
        return [$outputs];
    }

    protected function differentiate(array $dOutputs) : array
    {
        $K = $this->backend;
        $dInputs = $K->squeeze(
            $dOutputs[0],
            $this->axis,
        );
        return [$dInputs];
    }

}
