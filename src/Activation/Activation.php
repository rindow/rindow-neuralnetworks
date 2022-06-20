<?php
namespace Rindow\NeuralNetworks\Activation;

use Interop\Polite\Math\Matrix\NDArray;

interface Activation
{
    public function forward(object $status, NDArray $inputs, bool $training) : NDArray;
    public function backward(object $status, NDArray $dOutputs) : NDArray;
}
