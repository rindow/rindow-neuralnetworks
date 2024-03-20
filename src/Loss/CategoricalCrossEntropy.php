<?php
namespace Rindow\NeuralNetworks\Loss;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use ArrayAccess;

class CategoricalCrossEntropy extends AbstractLoss
{
    protected function call(NDArray $trues, NDArray $predicts) : NDArray
    {
        $K = $this->backend;
        if($this->fromLogits) {
            $predicts = $K->softmax($predicts);
        }
        [$trues,$predicts] = $this->flattenShapes($trues,$predicts);
        $container = $this->container();
        $container->trues = $trues;
        $container->predicts = $predicts;
        $outputs = $K->categoricalCrossEntropy(
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
        $dInputs = $K->dCategoricalCrossEntropy(
            $dLoss, $container->trues, $container->predicts,
            $this->fromLogits, $this->reduction
        );
        $dInputs = $this->reshapePredicts($dInputs);
        return [$dInputs];
    }

    public function accuracy(
        NDArray $trues, NDArray $predicts) : float
    {
        $K = $this->backend;
        [$trues,$predicts] = $this->flattenShapes($trues,$predicts);
        if($this->fromLogits) {
            //$predicts = $this->activationFunction($predicts);
            $predicts = $K->softmax($predicts);
        }
        $ndim = $trues->ndim();
        if($ndim>2){
            $shape = $trues->shape();
            $inputDim = array_pop($shape);
            $batch = array_product($shape);
            $trues = $trues->reshape([$batch,$inputDim]);
            $predicts = $predicts->reshape([$batch,$inputDim]);
        }
        // calc accuracy
        if($predicts->shape()[1]>2147483648) {
            $dtype = NDArray::int64;
        } else {
            $dtype = NDArray::int32;
        }
        $predicts = $K->argmax($predicts, axis:1, dtype:$dtype);
        $trues = $K->argmax($trues, axis:1, dtype:$dtype);
        $sum = $K->sum($K->equal($trues, $predicts));
        $sum = $K->scalar($sum);
        $accuracy = $sum/$trues->shape()[0];
        return $accuracy;
    }

    public function accuracyMetric() : string
    {
        return 'categorical_accuracy';
    }
}
