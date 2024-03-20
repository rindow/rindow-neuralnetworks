<?php
namespace Rindow\NeuralNetworks\Loss;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use DomainException;
use ArrayAccess;

class MeanSquaredError extends AbstractLoss
{
    public function __construct(
        object $backend,
        string $reduction=null,
    )
    {
        parent::__construct($backend,from_logits:null,reduction:$reduction);
    }

    protected function call(NDArray $trues, NDArray $predicts) : NDArray
    {
        $K = $this->backend;
        [$trues,$predicts] = $this->flattenShapes($trues,$predicts);
        $container = $this->container();
        //$this->assertOutputShape($predicts);
        //if($trues->ndim()!=2) {
        //    throw new InvalidArgumentException('categorical\'s "trues" must be shape of [batchsize,1].');
        //}
        $container->trues = $trues;
        $container->predicts = $predicts;
        $outputs = $K->meanSquaredError($trues, $predicts, reduction:$this->reduction);
        $outputs = $this->reshapeLoss($outputs);
        return $outputs;
    }

    protected function differentiate(array $dOutputs, ArrayAccess $grads=null, array $oidsToCollect=null) : array
    {
        $K = $this->backend;
        $dLoss = $this->flattenLoss($dOutputs[0]);
        $container = $this->container();
        $dInputs = $K->dMeanSquaredError(
            $dLoss, $container->trues, $container->predicts, reduction:$this->reduction);
        $dInputs = $this->reshapePredicts($dInputs);
        return [$dInputs];
    }

    public function accuracy(
        NDArray $trues, NDArray $predicts) : float
    {
        $K = $this->backend;
        [$trues,$predicts] = $this->flattenShapes($trues,$predicts);
        // calc accuracy
        $shape=$predicts->shape();
        if(count($shape)>=2) {
            if($predicts->shape()[1]>2147483648) {
                $dtype = NDArray::int64;
            } else {
                $dtype = NDArray::int32;
            }
            $predicts = $K->argmax($predicts, axis:1, dtype:$dtype);
            $trues = $K->argmax($trues, axis:1, dtype:$dtype);
            $sum = $K->sum($K->equal($trues, $predicts));
        } else {
            $sum = $K->nrm2($K->sub($predicts,$trues));
        }
        $sum = $K->scalar($sum);
        $accuracy = $sum/$trues->shape()[0];
        return $accuracy;
    }

    public function accuracyMetric() : string
    {
        return 'categorical_accuracy';
    }
}
