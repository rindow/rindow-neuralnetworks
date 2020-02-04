<?php
namespace Rindow\NeuralNetworks\Loss;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Layer\Layer;
use Rindow\NeuralNetworks\Layer\AbstractLayer;
use InvalidArgumentException;
use DomainException;

class SoftmaxWithSparseCategoricalCrossEntropy extends AbstractLayer implements Layer,LossLayer
{
    protected $backend;
    protected $outputs;
    protected $trues;

    public function __construct($backend,array $options=null)
    {
        //extract($this->extractArgs([
        //],$options));
        $this->backend = $K = $backend;
    }

    protected function call(NDArray $inputs, bool $training) : NDArray
    {
        $K = $this->backend;
        $this->outputs = $K->softmax($inputs);
        return $this->outputs;
    }

    public function loss(NDArray $trues, NDArray $predicts) : float
    {
        $K = $this->backend;
        $this->assertOutputShape($predicts);
        if($trues->ndim()!=1) {
            throw new InvalidArgumentException('categorical\'s "trues" must be shape of [batchsize,1].');
        }
        $this->trues = $trues;
        $this->loss = $K->sparseCrossEntropyError($trues, $predicts);
        return $this->loss;
    }

    public function differentiateLoss(float $loss=1) : NDArray
    {
        // Dummy to check the dOutputs in backward
        return $this->outputs;
    }

    protected function differentiate(NDArray $dummy) : NDArray
    {
        $K = $this->backend;
        return $K->dSoftmaxSparseCrossEntropyError($this->trues,$this->outputs);
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
