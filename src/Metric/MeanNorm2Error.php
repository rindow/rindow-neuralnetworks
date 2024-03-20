<?php
namespace Rindow\NeuralNetworks\Metric;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

class MeanNorm2Error extends AbstractMetric
{
    protected string $name = 'mean_norm2_error';

    public function forward(NDArray $trues, NDArray $predicts) : float
    {
        $K = $this->backend;
        $error = $K->scalar($K->nrm2($K->sub($predicts,$trues)));
        return $error/$trues->size();
    }
}
