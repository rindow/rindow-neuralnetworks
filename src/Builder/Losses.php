<?php
namespace Rindow\NeuralNetworks\Builder;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Loss\SoftmaxWithSparseCategoricalCrossEntropy;
use Rindow\NeuralNetworks\Loss\MeanSquaredError;

class Losses
{
    protected $backend;

    public function __construct($backend)
    {
        $this->backend = $backend;
    }

    public function MeanSquaredError(array $options=null)
    {
        return new MeanSquaredError($this->backend,$options);
    }

    public function SoftmaxWithSparseCategoricalCrossEntropy(array $options=null)
    {
        return new SoftmaxWithSparseCategoricalCrossEntropy($this->backend,$options);
    }

    public function SigmoidWithCrossEntropyError(array $options=null)
    {
        return new SigmoidWithCrossEntropyError($this->backend,$options);
    }
}
