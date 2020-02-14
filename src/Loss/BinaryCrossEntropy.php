<?php
namespace Rindow\NeuralNetworks\Loss;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

class BinaryCrossEntropy extends AbstractCrossEntropy
{
    protected function activationFunction(NDArray $predicts) : NDArray
    {
        return $this->backend->sigmoid($predicts);
    }

    protected function lossFunction(NDArray $trues, NDArray $predicts, bool $fromLogits) : float
    {
        if($predicts->ndim()!=2||$predicts->shape()[1]!=1) {
            throw new InvalidArgumentException('Invalid shape of predicts: ['.implode(',',$predicts->shape()).']');
        }
        $predicts = $predicts->reshape([$predicts->shape()[0]]);
        return $this->backend->categoricalCrossEntropy($trues, $predicts);
    }

    protected function deltaLossFunction(NDArray $trues, NDArray $predicts, bool $fromLogits) : NDArray
    {
        $newPredicts = $predicts->reshape([$predicts->shape()[0]]);
        $dx = $this->backend->dCategoricalCrossEntropy(
                                            $trues, $newPredicts, $fromLogits);
        return $dx->reshape($predicts->shape());
    }

    public function accuracy(
        NDArray $trues, NDArray $predicts) : float
    {
        $K = $this->backend;
        if($predicts->ndim()!=2||$predicts->shape()[1]!=1) {
            throw new InvalidArgumentException('Invalid shape of predicts: ['.implode(',',$predicts->shape()).']');
        }
        $predicts = $predicts->reshape([$predicts->shape()[0]]);
        if($trues->shape()!=$predicts->shape())
            throw new InvalidArgumentException('unmatch shape of trues and predicts results');
        // calc accuracy
        $predicts = $K->dmaximum($K->copy($predicts),0.5);
        if($trues->dtype()!=$predicts->dtype()) {
            $predicts = $K->cast($predicts,$trues->dtype());
        }
        $accuracy = $K->sum($K->equal($trues, $predicts))
                            / (float)$trues->shape()[0];
        return $accuracy;
    }
}
