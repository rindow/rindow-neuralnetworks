<?php
namespace Rindow\NeuralNetworks\Metric;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

class SparseCategoricalAccuracy extends AbstractMetric
{
    protected string $name = 'sparse_categorical_accuracy';

    public function forward(NDArray $trues, NDArray $predicts) : float
    {
        $K = $this->backend;
        [$trues,$predicts] = $this->flattenShapesForSparse($trues, $predicts);
        if(!$K->isInt($trues)) {
            throw new InvalidArgumentException('trues must be integers.');
        }
        $predicts = $K->argMax($predicts,axis:-1,dtype:$trues->dtype());
        $equals = $K->scalar($K->sum($K->equal($trues,$predicts)));
        return $equals/$trues->size();
    }
}
