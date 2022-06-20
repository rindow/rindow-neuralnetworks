<?php
namespace Rindow\NeuralNetworks\Builder;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Layer\Activation;
use Rindow\NeuralNetworks\Layer\Embedding;
use Rindow\NeuralNetworks\Layer\Dense;
use Rindow\NeuralNetworks\Layer\Input;
use Rindow\NeuralNetworks\Layer\Flatten;
use Rindow\NeuralNetworks\Layer\ExpandDims;
use Rindow\NeuralNetworks\Layer\RepeatVector;
use Rindow\NeuralNetworks\Layer\Concatenate;
use Rindow\NeuralNetworks\Layer\Max;
use Rindow\NeuralNetworks\Layer\Gather;
use Rindow\NeuralNetworks\Layer\Conv1D;
use Rindow\NeuralNetworks\Layer\Conv2D;
use Rindow\NeuralNetworks\Layer\Conv3D;
use Rindow\NeuralNetworks\Layer\MaxPooling1D;
use Rindow\NeuralNetworks\Layer\MaxPooling2D;
use Rindow\NeuralNetworks\Layer\MaxPooling3D;
use Rindow\NeuralNetworks\Layer\AveragePooling1D;
use Rindow\NeuralNetworks\Layer\AveragePooling2D;
use Rindow\NeuralNetworks\Layer\AveragePooling3D;
use Rindow\NeuralNetworks\Layer\GlobalMaxPooling1D;
use Rindow\NeuralNetworks\Layer\GlobalMaxPooling2D;
use Rindow\NeuralNetworks\Layer\GlobalMaxPooling3D;
use Rindow\NeuralNetworks\Layer\GlobalAveragePooling1D;
use Rindow\NeuralNetworks\Layer\GlobalAveragePooling2D;
use Rindow\NeuralNetworks\Layer\GlobalAveragePooling3D;
use Rindow\NeuralNetworks\Layer\Dropout;
use Rindow\NeuralNetworks\Layer\BatchNormalization;
use Rindow\NeuralNetworks\Layer\SimpleRNN;
use Rindow\NeuralNetworks\Layer\LSTM;
use Rindow\NeuralNetworks\Layer\GRU;
use Rindow\NeuralNetworks\Layer\Attention;

class Layers
{
    protected $backend;

    public function __construct(object $backend)
    {
        $this->backend = $backend;
    }

    public function Activation(
        $activation, ...$options)
    {
        return new Activation($this->backend,$activation,...$options);
    }

    public function Dense(int $units, ...$options)
    {
        return new Dense($this->backend, $units, ...$options);
    }

    public function Input(
        ...$options)
    {
        return new Input($this->backend, ...$options);
    }

    public function Flatten(
        ...$options)
    {
        return new Flatten($this->backend, ...$options);
    }

    public function ExpandDims(
        int $axis,
        ...$options)
    {
        return new ExpandDims($this->backend, $axis, ...$options);
    }

    public function RepeatVector(
        int $repeats,
        ...$options)
    {
        return new RepeatVector($this->backend, $repeats, ...$options);
    }

    public function Concatenate(
        ...$options)
    {
        return new Concatenate($this->backend, ...$options);
    }

    public function Max(
        ...$options)
    {
        return new Max($this->backend, ...$options);
    }

    public function Gather(
        ...$options)
    {
        return new Gather($this->backend, ...$options);
    }

    public function Conv1D(
        int $filters, int|array $kernel_size, ...$options)
    {
        return new Conv1D(
            $this->backend,
            $filters,
            $kernel_size,
            ...$options);
    }

    public function Conv2D(
        int $filters, int|array $kernel_size, ...$options)
    {
        return new Conv2D(
            $this->backend,
            $filters,
            $kernel_size,
            ...$options);
    }

    public function Conv3D(
        int $filters, int|array $kernel_size, ...$options)
    {
        return new Conv3D(
            $this->backend,
            $filters,
            $kernel_size,
            ...$options);
    }

    public function MaxPooling1D(
        ...$options)
    {
        return new MaxPooling1D(
            $this->backend,
            ...$options);
    }

    public function MaxPooling2D(
        ...$options)
    {
        return new MaxPooling2D(
            $this->backend,
            ...$options);
    }

    public function MaxPooling3D(
        ...$options)
    {
        return new MaxPooling3D(
            $this->backend,
            ...$options);
    }

    public function AveragePooling1D(
        ...$options)
    {
        return new AveragePooling1D(
            $this->backend,
            ...$options);
    }

    public function AveragePooling2D(
        ...$options)
    {
        return new AveragePooling2D(
            $this->backend,
            ...$options);
    }

    public function AveragePooling3D(
        ...$options)
    {
        return new AveragePooling3D(
            $this->backend,
            ...$options);
    }

    public function GlobalMaxPooling1D(
        ...$options)
    {
        return new GlobalMaxPooling1D(
            $this->backend,
            ...$options);
    }

    public function GlobalMaxPooling2D(
        ...$options)
    {
        return new GlobalMaxPooling2D(
            $this->backend,
            ...$options);
    }

    public function GlobalMaxPooling3D(
        ...$options)
    {
        return new GlobalMaxPooling3D(
            $this->backend,
            ...$options);
    }

    public function GlobalAveragePooling1D(
        ...$options)
    {
        return new GlobalAveragePooling1D(
            $this->backend,
            ...$options);
    }

    public function GlobalAveragePooling2D(
        ...$options)
    {
        return new GlobalAveragePooling2D(
            $this->backend,
            ...$options);
    }

    public function GlobalAveragePooling3D(
        ...$options)
    {
        return new GlobalAveragePooling3D(
            $this->backend,
            ...$options);
    }

    public function Dropout(
        float $rate, ...$options)
    {
        return new Dropout($this->backend,$rate,...$options);
    }

    public function BatchNormalization(...$options)
    {
        return new BatchNormalization($this->backend,...$options);
    }

    public function Embedding(int $inputDim,int $outputDim, ...$options)
    {
        return new Embedding($this->backend, $inputDim, $outputDim, ...$options);
    }

    public function SimpleRNN(int $units, ...$options)
    {
        return new SimpleRNN($this->backend, $units, ...$options);
    }

    public function LSTM(int $units, ...$options)
    {
        return new LSTM($this->backend, $units, ...$options);
    }

    public function GRU(int $units, ...$options)
    {
        return new GRU($this->backend, $units, ...$options);
    }

    public function Attention(...$options)
    {
        return new Attention($this->backend, ...$options);
    }
}
