<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;

class Tanh extends AbstractFunction
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

        // 前向计算：tanh(x)
        $output = $K->tanh($x); // Backend::tanh(NDArray): NDArray
        $container->output = $output; // 保存输出以供反向传播使用

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
        $output = $container->output; // 获取前向计算的输出
        $dOutput = $dOutputs[0];

        // 反向传播：grad = dOutput * (1 - tanh(x)^2)
        $grad = $K->mul( // mul(NDArray, NDArray): NDArray
            $dOutput,
            $K->sub( // sub(NDArray, NDArray): NDArray
                $K->onesLike($output), // onesLike(NDArray): NDArray
                $K->square($output) // square(NDArray): NDArray
            )
        );

        return [$grad];
    }
}