<?php
namespace Rindow\NeuralNetworks\Metric;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

class MeanSquaredError extends AbstractMetric
{
    protected string $name = 'mse';

    public function forward(NDArray $trues, NDArray $predicts) : float
    {
        $K = $this->backend;
        $error = $K->scalar($K->sum($K->square($K->sub($predicts,$trues))));
        return $error/$trues->size();
    }
}
