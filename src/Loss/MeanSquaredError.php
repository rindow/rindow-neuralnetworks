<?php
namespace Rindow\NeuralNetworks\Loss;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use DomainException;
use ArrayAccess;

class MeanSquaredError extends AbstractLoss implements Loss
{
    protected $backend;
    //protected $trues;
    //protected $predicts;

    public function __construct(object $backend)
    {
        $this->backend = $backend;
    }

    public function getConfig() : array
    {
        return [
        ];
    }

    protected function call(NDArray $trues, NDArray $predicts) : NDArray
    {
        $K = $this->backend;
        $container = $this->container();
        //$this->assertOutputShape($predicts);
        //if($trues->ndim()!=2) {
        //    throw new InvalidArgumentException('categorical\'s "trues" must be shape of [batchsize,1].');
        //}
        if($trues->shape()!=$predicts->shape())
            throw new InvalidArgumentException('unmatch shape of trues and predicts results');
        $container->trues = $trues;
        $container->predicts = $predicts;
        $loss = $K->meanSquaredError($trues, $predicts);
        return $loss;
    }

    protected function differentiate(array $dOutputs, ArrayAccess $grads=null, array $oidsToCollect=null) : array
    {
        $K = $this->backend;
        $container = $this->container();
        $dInputs = $K->dMeanSquaredError($container->trues, $container->predicts);
        return [$dInputs];
    }

    public function accuracy(
        NDArray $trues, NDArray $predicts) : float
    {
        $K = $this->backend;
        if($trues->shape()!=$predicts->shape())
            throw new InvalidArgumentException('unmatch shape of trues and predicts results');
        // calc accuracy
        $shape=$predicts->shape();
        if(count($shape)>=2) {
            if($predicts->shape()[1]>2147483648) {
                $dtype = NDArray::int64;
            } else {
                $dtype = NDArray::int32;
            }
            $predicts = $K->argmax($predicts, $axis=1,$dtype);
            $trues = $K->argmax($trues, $axis=1,$dtype);
            $sum = $K->sum($K->equal($trues, $predicts));
        } else {
            $sum = $K->nrm2($K->sub($predicts,$trues));
        }
        $sum = $K->scalar($sum);
        $accuracy = $sum/$trues->shape()[0];
        return $accuracy;
    }
}
