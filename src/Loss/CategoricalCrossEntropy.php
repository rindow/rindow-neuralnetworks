<?php
namespace Rindow\NeuralNetworks\Loss;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

class CategoricalCrossEntropy extends AbstractCrossEntropy
{
    protected function activationFunction(NDArray $predicts) : NDArray
    {
        return $this->backend->softmax($predicts);
    }

    protected function lossFunction(NDArray $trues, NDArray $predicts, bool $fromLogits) : float
    {
        return $this->backend->categoricalCrossEntropy($trues, $predicts);
    }

    protected function deltaLossFunction(NDArray $trues, NDArray $predicts, bool $fromLogits) : NDArray
    {
        return $this->backend->dCategoricalCrossEntropy(
                                            $trues, $predicts, $fromLogits);
    }

    public function accuracy(
        NDArray $trues, NDArray $predicts) : float
    {
        $K = $this->backend;
        if($trues->shape()!=$predicts->shape())
            throw new InvalidArgumentException('unmatch shape of trues and predicts results');
        // calc accuracy
        if($predicts->shape()[1]>2147483648) {
            $dtype = NDArray::int64;
        } else {
            $dtype = NDArray::int32;
        }
        $predicts = $K->argmax($predicts, $axis=1,$dtype);
        $trues = $K->argmax($trues, $axis=1,$dtype);
        $accuracy = $K->sum($K->equal($trues, $predicts))
                            / (float)$trues->shape()[0];
        return $accuracy;
    }
}
