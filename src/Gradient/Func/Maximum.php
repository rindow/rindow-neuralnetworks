<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;

class Maximum extends AbstractFunction
{
    protected int $numOfInputs = 2;

    protected function call(array $inputs): array
    {
        $container = $this->container();
        $container->inputs = $inputs;

        $x = $inputs[0];
        $y = $inputs[1];
        $K = $this->backend;

        // max(x, y) = x if x > y else y
        $diff = $K->sub($x, $y); // sub(NDArray, NDArray)
        $mask = $K->greater($diff, 0); // greater(NDArray, float)
        $output = $K->add( // add(NDArray, NDArray)
            $K->mul($x, $mask), // mul(NDArray, NDArray)
            $K->mul($y, $K->sub($K->onesLike($mask), $mask)) // mul/sub
        );

        return [$output];
    }

    protected function differentiate(array $dOutputs): array
    {
        $K = $this->backend;
        $container = $this->container();
        [$x, $y] = $container->inputs;
        $dOutput = $dOutputs[0];

        $diff = $K->sub($x, $y); // sub(NDArray, NDArray)
        $mask = $K->greater($diff, 0); // greater(NDArray, float)
        $dx = $K->mul($dOutput, $mask); // mul(NDArray, NDArray)
        $dy = $K->mul($dOutput, $K->sub($K->onesLike($mask), $mask)); // mul/sub

        if ($x->ndim() != $dx->ndim()) {
            $dx = $K->sum($dx, axis: 0); // sum(NDArray, int)
        }
        if ($y->ndim() != $dy->ndim()) {
            $dy = $K->sum($dy, axis: 0); // sum(NDArray, int)
        }

        return [$dx, $dy];
    }
}