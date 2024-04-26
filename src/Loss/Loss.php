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

    /**
     * @param array<NDArray> $dOutputs
     * @param ArrayAccess<object,object> $grads
     * @param array<NDArray> $oidsToCollect
     * @return array<NDArray>
     */
    public function backward(array $dOutputs, ArrayAccess $grads=null, array $oidsToCollect=null) : array;

    public function accuracyMetric() : string;

    /**
     * @return array<string,mixed>
     */
    public function getConfig() : array;
}
