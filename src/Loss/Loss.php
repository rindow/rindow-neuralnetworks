<?php
namespace Rindow\NeuralNetworks\Loss;

use Interop\Polite\Math\Matrix\NDArray;
use ArrayAccess;

/**
 *
 */
interface Loss
{
    public function forward(NDArray $true, NDArray $x) : NDArray;
    public function backward(array $dOutputs, ArrayAccess $grads=null, array $oidsToCollect=null) : array;
    public function accuracy(NDArray $c_true, NDArray $y_pred) : float;
    public function accuracyMetric() : string;
    public function getConfig() : array;
}
