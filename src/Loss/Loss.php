<?php
namespace Rindow\NeuralNetworks\Loss;

use Interop\Polite\Math\Matrix\NDArray;

/**
 *
 */
interface Loss
{
    //public function loss(NDArray $true, NDArray $x) : float;
    //public function differentiateLoss() : NDArray;
    public function forward(NDArray $true, NDArray $x) : float;
    public function backward(array $dOutputs) : array;
    public function accuracy(NDArray $c_true, NDArray $y_pred) : float;
    public function getConfig() : array;
}
