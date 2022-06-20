<?php
namespace Rindow\NeuralNetworks\Loss;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

class BinaryCrossEntropy extends AbstractCrossEntropy
{
    protected function activationFunction(NDArray $inputs) : NDArray
    {
        $y = $this->backend->sigmoid($inputs);
        return $y;
    }

    protected function diffActivationFunction(NDArray $dOutputs, NDArray $outputs) : NDArray
    {
        return $this->backend->dSigmoid($dOutputs, $outputs);
    }

    protected function lossFunction(NDArray $trues, NDArray $predicts, bool $fromLogits) : NDArray
    {
        $ndimP = $predicts->ndim();
        if($ndimP!=1 && ($ndimP!=2 || $predicts->shape()[1]!=1)) {
            throw new InvalidArgumentException('Invalid shape of predicts: ['.implode(',',$predicts->shape()).']');
        }
        $ndimT = $trues->ndim();
        if($ndimT==1) {
            ;
        } elseif($ndimT==2 && $trues->shape()[1]==1) {
            $trues = $trues->reshape([$trues->shape()[0]]);
        } else {
            throw new InvalidArgumentException('Invalid shape of trues: ['.implode(',',$trues->shape()).']');
        }
        $predicts = $predicts->reshape([$predicts->shape()[0]]);
        #return $this->backend->categoricalCrossEntropy($trues, $predicts);
        return $this->backend->binaryCrossEntropy($trues, $predicts);
    }

    protected function diffLossFunction(NDArray $trues, NDArray $predicts, bool $fromLogits) : NDArray
    {
        $newPredicts = $predicts->reshape([$predicts->shape()[0]]);
        $dx = $this->backend->dBinaryCrossEntropy(
                                            $trues, $newPredicts, $fromLogits);
        return $dx->reshape($predicts->shape());
    }

    public function accuracy(
        NDArray $trues, NDArray $predicts) : float
    {
        $K = $this->backend;
        $shape = $predicts->shape();
        $inputDim = array_pop($shape);
        if($inputDim!=1) {
            throw new InvalidArgumentException('Invalid shape of predicts: ['.implode(',',$predicts->shape()).']');
        }
        if($trues->shape()!=$shape)
            throw new InvalidArgumentException('unmatch shape of trues and predicts results');
        if($this->fromLogits) {
            $predicts = $this->activationFunction($predicts);
        }
        $predicts = $predicts->reshape([$predicts->size()]);
        // calc accuracy
        $predicts = $K->greater($K->copy($predicts),0.5);
        if($trues->dtype()!=$predicts->dtype()) {
            $predicts = $K->cast($predicts,$trues->dtype());
        }
        $sum = $K->sum($K->equal($trues, $predicts));
        $sum = $K->scalar($sum);
        $accuracy = $sum/$trues->size();
        return $accuracy;
    }
}
