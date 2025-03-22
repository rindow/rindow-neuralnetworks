<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;

class Minimum extends AbstractFunction
{
    protected int $numOfInputs = 2;

    /**
     * @param array<NDArray> $inputs
     * @return array<NDArray>
     */
    protected function call(array $inputs): array
    {
        $container = $this->container();
        $container->inputs = $inputs;

        $x = $inputs[0];
        $y = $inputs[1];
        $K = $this->backend;

        // 计算 x - y
        $diff = $K->sub($x, $y);           // sub(NDArray, NDArray): NDArray
        // mask = (x < y) = (x - y < 0)
        $mask = $K->less($diff, 0);        // less(NDArray, float): NDArray
        // output = x * mask + y * (1 - mask)
        $output = $K->add(                  // add(NDArray, NDArray): NDArray
            $K->mul($x, $mask),             // mul(NDArray, NDArray): NDArray
            $K->mul($y, $K->sub(            // mul(NDArray, NDArray): NDArray
                $K->onesLike($mask), $mask  // sub(NDArray, NDArray): NDArray
            ))
        );

        return [$output];
    }

    /**
     * @param array<NDArray> $dOutputs
     * @return array<NDArray>
     */
    protected function differentiate(array $dOutputs): array
    {
        $K = $this->backend;
        $container = $this->container();
        [$x, $y] = $container->inputs;
        $dOutput = $dOutputs[0];

        // 计算 x - y
        $diff = $K->sub($x, $y);           // sub(NDArray, NDArray): NDArray
        // mask = (x < y) = (x - y < 0)
        $mask = $K->less($diff, 0);        // less(NDArray, float): NDArray
        // dx = dOutput * (x < y)
        $dx = $K->mul($dOutput, $mask);    // mul(NDArray, NDArray): NDArray
        // dy = dOutput * (1 - (x < y))
        $dy = $K->mul($dOutput, $K->sub(   // mul(NDArray, NDArray): NDArray
            $K->onesLike($mask), $mask     // sub(NDArray, NDArray): NDArray
        ));

        // 处理广播情况
        if ($x->ndim() != $dx->ndim()) {
            $dx = $K->sum($dx, axis: 0);   // sum(NDArray, int): NDArray
        }
        if ($y->ndim() != $dy->ndim()) {
            $dy = $K->sum($dy, axis: 0);   // sum(NDArray, int): NDArray
        }

        return [$dx, $dy];
    }
}