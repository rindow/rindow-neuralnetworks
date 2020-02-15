<?php
namespace Rindow\NeuralNetworks\Loss;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

class SparseCategoricalCrossEntropy extends AbstractCrossEntropy
{
    protected function activationFunction(NDArray $inputs) : NDArray
    {
        return $this->backend->softmax($inputs);
    }

    protected function diffActivationFunction(NDArray $dOutputs, NDArray $outputs) : NDArray
    {
        return $this->backend->dSoftmax($dOutputs, $outputs);
    }

    protected function lossFunction(NDArray $trues, NDArray $predicts, bool $fromLogits) : float
    {
        return $this->backend->sparseCategoricalCrossEntropy($trues, $predicts);
    }

    protected function diffLossFunction(NDArray $trues, NDArray $predicts, bool $fromLogits) : NDArray
    {
        return $this->backend->dSparseCategoricalCrossEntropy(
                                            $trues, $predicts, $fromLogits);
    }

    public function accuracy(
        NDArray $c_true, NDArray $y_pred) : float
    {
        $K = $this->backend;
        // transrate one hot to categorical labels
        $c_pred = $K->argmax($y_pred, $axis=1,$c_true->dtype());
        if($c_true->shape()!=$c_pred->shape())
            throw new InvalidArgumentException('unmatch categorical true and predict results');
        // calc accuracy
        $accuracy = $K->sum($K->equal($c_true, $c_pred))
                            / (float)$c_true->shape()[0];
        return $accuracy;
    }
}
