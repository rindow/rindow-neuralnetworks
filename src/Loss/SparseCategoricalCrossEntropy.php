<?php
namespace Rindow\NeuralNetworks\Loss;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use ArrayAccess;

class SparseCategoricalCrossEntropy extends AbstractLoss
{
    protected function call(NDArray $trues, NDArray $predicts) : NDArray
    {
        $K = $this->backend;
        if(!$K->isInt($trues)) {
            throw new InvalidArgumentException('trues must be integers.');
        }
        [$trues,$predicts] = $this->flattenShapesForSparse($trues,$predicts);
        if($this->fromLogits) {
            $predicts = $K->softmax($predicts);
        }
        $container = $this->container();
        $container->trues = $trues;
        $container->predicts = $predicts;
        $outputs = $K->sparseCategoricalCrossEntropy(
            $trues, $predicts,
            $this->fromLogits, $this->reduction
        );
        $outputs = $this->reshapeLoss($outputs);
        return $outputs;
    }

    protected function differentiate(array $dOutputs, ArrayAccess $grads=null, array $oidsToCollect=null) : array
    {
        $K = $this->backend;
        $dLoss = $this->flattenLoss($dOutputs[0]);
        $container = $this->container();
        $dInputs = $K->dSparseCategoricalCrossEntropy(
            $dLoss, $container->trues, $container->predicts,
            $this->fromLogits, $this->reduction
        );
        $dInputs = $this->reshapePredicts($dInputs);
        return [$dInputs];
    }

    public function accuracy(
        NDArray $c_true, NDArray $y_pred) : float
    {
        $K = $this->backend;
        [$c_true,$y_pred] = $this->flattenShapesForSparse($c_true,$y_pred);
        // transrate one hot to categorical labels
        if($this->fromLogits) {
            //$y_pred = $this->activationFunction($y_pred);
            $y_pred = $K->softmax($y_pred);
        }
        $ndim = $c_true->ndim();
        if($ndim>1){
            $c_true = $c_true->reshape([$c_true->size()]);
            $predShape = $y_pred->shape();
            $inputDim = array_pop($predShape);
            $y_pred = $y_pred->reshape([array_product($predShape),$inputDim]);
        }
        $c_pred = $K->argMax($y_pred, axis:1, dtype:$c_true->dtype());
        if($c_true->shape()!=$c_pred->shape())
            throw new InvalidArgumentException('unmatch categorical true and predict results');
        // calc accuracy
        $sum = $K->sum($K->equal($c_true, $c_pred));
        $sum = $K->scalar($sum);
        $accuracy = $sum/$c_true->shape()[0];
        return $accuracy;
    }

    public function accuracyMetric() : string
    {
        return 'sparse_categorical_accuracy';
    }
}
