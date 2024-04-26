<?php
namespace Rindow\NeuralNetworks\Gradient;

use Interop\Polite\Math\Matrix\NDArray;

interface GraphFunction
{
    /**
     * @param array<NDArray> $inputs
     * @param array<string,mixed> $options
     * @return array<NDArray>
     */
    public function _rawCall(array $inputs,array $options) : array;
}