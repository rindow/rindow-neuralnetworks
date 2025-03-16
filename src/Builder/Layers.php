<?php
namespace Rindow\NeuralNetworks\Builder;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Activation\Activation as ActivationInterface;

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
use Rindow\NeuralNetworks\Layer\LayerNormalization;
use Rindow\NeuralNetworks\Layer\SimpleRNN;
use Rindow\NeuralNetworks\Layer\LSTM;
use Rindow\NeuralNetworks\Layer\GRU;
use Rindow\NeuralNetworks\Layer\Attention;
use Rindow\NeuralNetworks\Layer\MultiHeadAttention;
use Rindow\NeuralNetworks\Layer\InheritMask;
use Rindow\NeuralNetworks\Layer\Add;
use Rindow\NeuralNetworks\Layer\Debug;

class Layers
{
    protected object $backend;

    public function __construct(object $backend)
    {
        $this->backend = $backend;
    }

    public function Debug(mixed ...$options) : object
    {
        return new Debug($this->backend,...$options);
    }

    public function Activation(
        string|ActivationInterface $activation, mixed ...$options) : object
    {
        return new Activation($this->backend,$activation,...$options);
    }

    public function Dense(int $units, mixed ...$options) : object
    {
        return new Dense($this->backend, $units, ...$options);
    }

    public function Input(
        mixed ...$options) : object
    {
        return new Input($this->backend, ...$options);
    }

    public function Flatten(
        mixed ...$options) : object
    {
        return new Flatten($this->backend, ...$options);
    }

    public function ExpandDims(
        int $axis,
        mixed ...$options) : object
    {
        return new ExpandDims($this->backend, $axis, ...$options);
    }

    public function RepeatVector(
        int $repeats,
        mixed ...$options) : object
    {
        return new RepeatVector($this->backend, $repeats, ...$options);
    }

    public function Concatenate(
        mixed ...$options) : object
    {
        return new Concatenate($this->backend, ...$options);
    }

    public function Max(
        mixed ...$options) : object
    {
        return new Max($this->backend, ...$options);
    }

    public function Gather(
        mixed ...$options) : object
    {
        return new Gather($this->backend, ...$options);
    }

    /**
     * @param int|array<int> $kernel_size
     */
    public function Conv1D(
        int $filters, int|array $kernel_size, mixed ...$options) : object
    {
        return new Conv1D(
            $this->backend,
            $filters,
            $kernel_size,
            ...$options);
    }

    /**
     * @param int|array<int> $kernel_size
     */
    public function Conv2D(
        int $filters, int|array $kernel_size, mixed ...$options) : object
    {
        return new Conv2D(
            $this->backend,
            $filters,
            $kernel_size,
            ...$options);
    }

    /**
     * @param int|array<int> $kernel_size
     */
    public function Conv3D(
        int $filters, int|array $kernel_size, mixed ...$options) : object
    {
        return new Conv3D(
            $this->backend,
            $filters,
            $kernel_size,
            ...$options);
    }

    public function MaxPooling1D(
        mixed ...$options) : object
    {
        return new MaxPooling1D(
            $this->backend,
            ...$options);
    }

    public function MaxPooling2D(
        mixed ...$options) : object
    {
        return new MaxPooling2D(
            $this->backend,
            ...$options);
    }

    public function MaxPooling3D(
        mixed ...$options) : object
    {
        return new MaxPooling3D(
            $this->backend,
            ...$options);
    }

    public function AveragePooling1D(
        mixed ...$options) : object
    {
        return new AveragePooling1D(
            $this->backend,
            ...$options);
    }

    public function AveragePooling2D(
        mixed ...$options) : object
    {
        return new AveragePooling2D(
            $this->backend,
            ...$options);
    }

    public function AveragePooling3D(
        mixed ...$options) : object
    {
        return new AveragePooling3D(
            $this->backend,
            ...$options);
    }

    public function GlobalMaxPooling1D(
        mixed ...$options) : object
    {
        return new GlobalMaxPooling1D(
            $this->backend,
            ...$options);
    }

    public function GlobalMaxPooling2D(
        mixed ...$options) : object
    {
        return new GlobalMaxPooling2D(
            $this->backend,
            ...$options);
    }

    public function GlobalMaxPooling3D(
        mixed ...$options) : object
    {
        return new GlobalMaxPooling3D(
            $this->backend,
            ...$options);
    }

    public function GlobalAveragePooling1D(
        mixed ...$options) : object
    {
        return new GlobalAveragePooling1D(
            $this->backend,
            ...$options);
    }

    public function GlobalAveragePooling2D(
        mixed  ...$options) : object
    {
        return new GlobalAveragePooling2D(
            $this->backend,
            ...$options);
    }

    public function GlobalAveragePooling3D(
        mixed ...$options) : object
    {
        return new GlobalAveragePooling3D(
            $this->backend,
            ...$options);
    }

    public function Dropout(
        float $rate, mixed ...$options) : object
    {
        return new Dropout($this->backend,$rate,...$options);
    }

    public function BatchNormalization(mixed ...$options) : object
    {
        return new BatchNormalization($this->backend,...$options);
    }

    public function LayerNormalization(mixed ...$options) : object
    {
        return new LayerNormalization($this->backend,...$options);
    }

    public function Embedding(int $inputDim,int $outputDim, mixed ...$options) : object
    {
        return new Embedding($this->backend, $inputDim, $outputDim, ...$options);
    }

    public function SimpleRNN(int $units, mixed ...$options) : object
    {
        return new SimpleRNN($this->backend, $units, ...$options);
    }

    public function LSTM(int $units, mixed ...$options) : object
    {
        return new LSTM($this->backend, $units, ...$options);
    }

    public function GRU(int $units, mixed ...$options) : object
    {
        return new GRU($this->backend, $units, ...$options);
    }

    public function Attention(mixed ...$options) : object
    {
        return new Attention($this->backend, ...$options);
    }

    public function MultiHeadAttention(mixed ...$options) : object
    {
        return new MultiHeadAttention($this->backend, ...$options);
    }

    public function InheritMask(mixed ...$options) : object
    {
        return new InheritMask($this->backend, ...$options);
    }
    
    public function Add(mixed ...$options) : object
    {
        return new Add($this->backend, ...$options);
    }
}
