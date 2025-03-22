<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;

class Where extends AbstractFunction
{
    protected int $numOfInputs = 3;

    /**
     * @param array<NDArray> $inputs
     * @return array<NDArray>
     */
    protected function call(array $inputs): array
    {
        $container = $this->container();
        $container->inputs = $inputs;

        $condition = $inputs[0]; // 布尔 NDArray
        $x = $inputs[1];         // 当 condition 为真时选择
        $y = $inputs[2];         // 当 condition 为假时选择
        $K = $this->backend;

        // 前向计算：output = condition * x + (1 - condition) * y
        $mask = $condition; // 假设 condition 是 0/1 的布尔数组
        $output = $K->add( // add(NDArray, NDArray)
            $K->mul($x, $mask), // mul(NDArray, NDArray)
            $K->mul($y, $K->sub($K->onesLike($mask), $mask)) // mul/sub(NDArray, NDArray)
        );

        $container->mask = $mask; // 保存 mask 以供反向传播
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
        $mask = $container->mask; // 前向计算时的 mask
        $dOutput = $dOutputs[0];

        // 反向传播
        // dCondition 不需要计算（假设 condition 是不可微的）
        $dX = $K->mul($dOutput, $mask); // mul(NDArray, NDArray)
        $dY = $K->mul($dOutput, $K->sub($K->onesLike($mask), $mask)); // mul/sub(NDArray, NDArray)

        // 处理广播
        $x = $container->inputs[1];
        $y = $container->inputs[2];
        if ($x->ndim() != $dX->ndim()) {
            $dX = $K->sum($dX, axis: 0); // sum(NDArray, int)
        }
        if ($y->ndim() != $dY->ndim()) {
            $dY = $K->sum($dY, axis: 0); // sum(NDArray, int)
        }

        return [null, $dX, $dY]; // condition 无梯度
    }
}