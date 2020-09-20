<?php
namespace Rindow\NeuralNetworks\Layer;

use Interop\Polite\Math\Matrix\NDArray;

/**
 *
 */
interface Layer extends LayerBase
{
    public function forward(NDArray $inputs, bool $training) : NDArray;
    public function backward(NDArray $dOutputs) : NDArray;
}