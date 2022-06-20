<?php
namespace Rindow\NeuralNetworks\Layer;

use Interop\Polite\Math\Matrix\NDArray;

/**
 *
 */
interface Layer extends LayerBase
{
    public function forward(object $inputs, bool $training);
    public function backward(array $dOutputs) : array;
}
