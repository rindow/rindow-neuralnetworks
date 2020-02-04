<?php
namespace Rindow\NeuralNetworks\Loss;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use DomainException;

class MeanSquaredError implements LossLayer
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

    public function getConfig() : array
    {
        return [
        ];
    }

    public function loss(NDArray $trues, NDArray $predicts) : float
    {
        $K = $this->backend;
        //$this->assertOutputShape($predicts);
        if($trues->ndim()!=2) {
            throw new InvalidArgumentException('categorical\'s "trues" must be shape of [batchsize,1].');
        }
        if($trues->shape()!=$predicts->shape())
            throw new InvalidArgumentException('unmatch shape of trues and predicts results');
        $this->trues = $trues;
        $this->predicts = $predicts;
        $this->loss = $K->meanSquaredError($trues, $predicts);
        return $this->loss;
    }

    public function differentiateLoss() : NDArray
    {
        $K = $this->backend;
        return $K->dMeanSquaredError($this->trues, $this->predicts);
    }

    public function accuracy(
        NDArray $trues, NDArray $predicts) : float
    {
        $K = $this->backend;
        if($trues->shape()!=$predicts->shape())
            throw new InvalidArgumentException('unmatch shape of trues and predicts results');
        // calc accuracy
        $accuracy = $K->sum($K->mul($trues, $predicts))
                            / (float)$trues->shape()[0];
        return $accuracy;
    }
}
