<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;

class Abs extends AbstractFunction
{
    protected int $numOfInputs = 1;

    /**
     * @param array<NDArray> $inputs
     * @return array<NDArray>
     */
    protected function call(array $inputs): array
    {
        $container = $this->container();
        $container->inputs = $inputs;

        $x = $inputs[0];
        $K = $this->backend;

        // 前向计算：abs(x)
        $output = $K->abs($x); // Backend::abs(NDArray): NDArray
        $container->output = $output; // 保存输出以供反向传播

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
        $x = $container->inputs[0];
        $dOutput = $dOutputs[0];

        // 反向传播：grad = dOutput * sign(x)
        $grad = $K->mul( // Backend::mul(NDArray, NDArray): NDArray
            $dOutput,
            $K->sign($x) // Backend::sign(NDArray): NDArray
        );

        return [$grad];
    }
}