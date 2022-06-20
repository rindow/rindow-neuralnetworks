<?php
namespace Rindow\NeuralNetworks\Activation;

use Interop\Polite\Math\Matrix\NDArray;

interface Activation
{
    public function forward(NDArray $inputs, bool $training) : NDArray;
    public function backward(NDArray $dOutputs) : NDArray;
}
