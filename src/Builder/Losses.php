<?php
namespace Rindow\NeuralNetworks\Builder;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Loss\MeanSquaredError;
use Rindow\NeuralNetworks\Loss\SparseCategoricalCrossEntropy;
use Rindow\NeuralNetworks\Loss\CategoricalCrossEntropy;
use Rindow\NeuralNetworks\Loss\BinaryCrossEntropy;
use Rindow\NeuralNetworks\Loss\Huber;

class Losses
{
    protected $backend;

    public function __construct(object $backend)
    {
        $this->backend = $backend;
    }

    public function MeanSquaredError(...$options)
    {
        return new MeanSquaredError($this->backend, ...$options);
    }

    public function SparseCategoricalCrossEntropy(...$options)
    {
        return new SparseCategoricalCrossEntropy($this->backend, ...$options);
    }

    public function CategoricalCrossEntropy(...$options)
    {
        return new CategoricalCrossEntropy($this->backend, ...$options);
    }

    public function BinaryCrossEntropy(...$options)
    {
        return new BinaryCrossEntropy($this->backend, ...$options);
    }

    public function Huber(...$options)
    {
        return new Huber($this->backend, ...$options);
    }
}
