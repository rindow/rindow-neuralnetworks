<?php
namespace Rindow\NeuralNetworks\Builder;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Loss\MeanSquaredError;
use Rindow\NeuralNetworks\Loss\SparseCategoricalCrossEntropy;
use Rindow\NeuralNetworks\Loss\CategoricalCrossEntropy;
use Rindow\NeuralNetworks\Loss\BinaryCrossEntropy;

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

    public function SparseCategoricalCrossEntropy(array $options=null)
    {
        return new SparseCategoricalCrossEntropy($this->backend,$options);
    }

    public function CategoricalCrossEntropy(array $options=null)
    {
        return new CategoricalCrossEntropy($this->backend,$options);
    }

    public function BinaryCrossEntropy(array $options=null)
    {
        return new BinaryCrossEntropy($this->backend,$options);
    }
}
