<?php
namespace Rindow\NeuralNetworks\Metric;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

abstract class AbstractMetric implements Metric
{
    protected object $backend;
    protected float $state = 0.0;
    protected int $count = 0;

    public function __construct(
        object $backend,
        )
    {
        $this->backend = $backend;
    }

    public function reset() : void
    {
        $this->state = 0.0;
        $this->count = 0;
    }

    public function result() : float
    {
        if($this->count==0) {
            return 0;
        }
        return $this->state / $this->count;
    }

    public function name() : string
    {
        return $this->name;
    }

    public function update(NDArray $trues, NDArray $predicts) : void
    {
        $this->state += $this->forward($trues, $predicts);
        $this->count++;
    }

    public function immediateUpdate(float $value) : void
    {
        $this->state += $value;
        $this->count++;
    }

    public function __invoke(...$args) : mixed
    {
        [$trues,$predicts] = $args;
        return $this->forward($trues, $predicts);
    }

    protected function flattenShapes(NDArray $trues, NDArray $predicts) : array
    {
        $origTrueShape = $trues->shape();
        $origPredictsShape = $predicts->shape();
        if($trues->ndim()<$predicts->ndim()) {
            $shape = $trues->shape();
            array_push($shape,1);
            $trues = $trues->reshape($shape);
        }
        if($trues->shape()!=$predicts->shape()){
            throw new InvalidArgumentException('trues and predicts must be same shape of dimensions. '.
                'trues,predicts are ['.implode(',',$origTrueShape).'],['.implode(',',$predicts->shape()).']');
        }
        if($predicts->ndim()==1) {
            $size = $predicts->size();
            $predicts = $predicts->reshape([1,$size]);
            $trues = $trues->reshape([1,$size]);
        }
        //$origPredictsShape = $predicts->shape();
        //$orgTruesShape = $trues->shape();
        $batchShape = $predicts->shape();
        $feature = array_pop($batchShape);
        $batchSize = array_product($batchShape);
        $trues = $trues->reshape([$batchSize,$feature]);
        $predicts = $predicts->reshape([$batchSize,$feature]);
        return [$trues,$predicts];
    }

    protected function flattenShapesForSparse(NDArray $trues, NDArray $predicts) : array
    {
        $origTrueShape = $trues->shape();
        $origPredictsShape = $trues->shape();

        $batchShape = $predicts->shape();
        $feature = array_pop($batchShape);
        $batchSize = array_product($batchShape);
        if($trues->shape()!=$batchShape){
            throw new InvalidArgumentException('trues and predicts must be same batch-shape of dimensions. '.
                'trues,predicts are ['.implode(',',$origTrueShape).'],['.implode(',',$origPredictsShape->shape()).']');
        }

        $trues = $trues->reshape([$batchSize]);
        $predicts = $predicts->reshape([$batchSize,$feature]);
        return [$trues,$predicts];
    }
}
