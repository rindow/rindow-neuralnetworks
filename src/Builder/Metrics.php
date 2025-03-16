<?php
namespace Rindow\NeuralNetworks\Builder;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Metric\ScalarMetric;
use Rindow\NeuralNetworks\Metric\GenericMetric;
use Rindow\NeuralNetworks\Metric\SparseCategoricalAccuracy;
use Rindow\NeuralNetworks\Metric\CategoricalAccuracy;
use Rindow\NeuralNetworks\Metric\BinaryAccuracy;
use Rindow\NeuralNetworks\Metric\MeanNorm2Error;
use Rindow\NeuralNetworks\Metric\MeanSquaredError;

class Metrics
{
    protected object $backend;

    public function __construct(object $backend)
    {
        $this->backend = $backend;
    }

    public function ScalarMetric(string $name) : object
    {
        return new ScalarMetric($this->backend, $name);
    }

    public function GenericMetric(callable $func, ?string $name=null) : object
    {
        return new GenericMetric($this->backend, $func, $name);
    }

    public function SparseCategoricalAccuracy(mixed ...$options) : object
    {
        return new SparseCategoricalAccuracy($this->backend, ...$options);
    }

    public function CategoricalAccuracy(mixed ...$options) : object
    {
        return new CategoricalAccuracy($this->backend, ...$options);
    }

    public function BinaryAccuracy(mixed ...$options) : object
    {
        return new BinaryAccuracy($this->backend, ...$options);
    }

    public function MeanSquaredError(mixed ...$options) : object
    {
        return new MeanSquaredError($this->backend, ...$options);
    }

    public function MeanNorm2Error(mixed ...$options) : object
    {
        return new MeanNorm2Error($this->backend, ...$options);
    }
}
