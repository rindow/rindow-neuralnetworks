<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;

class Sigmoid extends AbstractFunction
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

        // 前向计算：sigmoid(x) = 1 / (1 + exp(-x))
        $output = $K->sigmoid($x); // Backend::sigmoid(NDArray): NDArray
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
        $output = $container->output; // 前向输出
        $dOutput = $dOutputs[0];

        // 反向传播：grad = dOutput * sigmoid(x) * (1 - sigmoid(x))
        $grad = $K->mul( // Backend::mul(NDArray, NDArray): NDArray
            $dOutput,
            $K->mul( // Backend::mul(NDArray, NDArray): NDArray
                $output,
                $K->sub( // Backend::sub(NDArray, NDArray): NDArray
                    $K->onesLike($output), // Backend::onesLike(NDArray): NDArray
                    $output
                )
            )
        );

        return [$grad];
    }
}