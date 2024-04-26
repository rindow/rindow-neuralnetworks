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
    protected object $backend;

    public function __construct(object $backend)
    {
        $this->backend = $backend;
    }

    public function MeanSquaredError(mixed ...$options) : object
    {
        return new MeanSquaredError($this->backend, ...$options);
    }

    public function SparseCategoricalCrossEntropy(mixed ...$options) : object
    {
        return new SparseCategoricalCrossEntropy($this->backend, ...$options);
    }

    public function CategoricalCrossEntropy(mixed ...$options) : object
    {
        return new CategoricalCrossEntropy($this->backend, ...$options);
    }

    public function BinaryCrossEntropy(mixed ...$options) : object
    {
        return new BinaryCrossEntropy($this->backend, ...$options);
    }

    public function Huber(mixed ...$options) : object
    {
        return new Huber($this->backend, ...$options);
    }
}
