<?php
namespace Rindow\NeuralNetworks\Builder;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Layer\Activation;
use Rindow\NeuralNetworks\Layer\Embedding;
use Rindow\NeuralNetworks\Layer\Dense;
use Rindow\NeuralNetworks\Layer\Input;
use Rindow\NeuralNetworks\Layer\Flatten;
use Rindow\NeuralNetworks\Layer\RepeatVector;
use Rindow\NeuralNetworks\Layer\Concatenate;
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

    public function __construct($backend)
    {
        $this->backend = $backend;
    }

    public function Activation(
        $activation,array $options=null)
    {
        return new Activation($this->backend,$activation,$options);
    }

    public function Dense(int $units, array $options=null)
    {
        return new Dense($this->backend, $units, $options);
    }

    public function Input(
        array $options=null)
    {
        return new Input($this->backend, $options);
    }

    public function Flatten(
        array $options=null)
    {
        return new Flatten($this->backend, $options);
    }

    public function RepeatVector(
        int $repeats,
        array $options=null)
    {
        return new RepeatVector($this->backend, $repeats, $options);
    }

    public function Concatenate(
        array $options=null)
    {
        return new Concatenate($this->backend, $options);
    }

    public function Conv1D(
        int $filters, $kernel_size, array $options=null)
    {
        return new Conv1D(
            $this->backend,
            $filters,
            $kernel_size,
            $options);
    }

    public function Conv2D(
        int $filters, $kernel_size, array $options=null)
    {
        return new Conv2D(
            $this->backend,
            $filters,
            $kernel_size,
            $options);
    }

    public function Conv3D(
        int $filters, $kernel_size, array $options=null)
    {
        return new Conv3D(
            $this->backend,
            $filters,
            $kernel_size,
            $options);
    }

    public function MaxPooling1D(
        array $options=null)
    {
        return new MaxPooling1D(
            $this->backend,
            $options);
    }

    public function MaxPooling2D(
        array $options=null)
    {
        return new MaxPooling2D(
            $this->backend,
            $options);
    }

    public function MaxPooling3D(
        array $options=null)
    {
        return new MaxPooling3D(
            $this->backend,
            $options);
    }

    public function AveragePooling1D(
        array $options=null)
    {
        return new AveragePooling1D(
            $this->backend,
            $options);
    }

    public function AveragePooling2D(
        array $options=null)
    {
        return new AveragePooling2D(
            $this->backend,
            $options);
    }

    public function AveragePooling3D(
        array $options=null)
    {
        return new AveragePooling3D(
            $this->backend,
            $options);
    }

    public function GlobalMaxPooling1D(
        array $options=null)
    {
        return new GlobalMaxPooling1D(
            $this->backend,
            $options);
    }

    public function GlobalMaxPooling2D(
        array $options=null)
    {
        return new GlobalMaxPooling2D(
            $this->backend,
            $options);
    }

    public function GlobalMaxPooling3D(
        array $options=null)
    {
        return new GlobalMaxPooling3D(
            $this->backend,
            $options);
    }

    public function GlobalAveragePooling1D(
        array $options=null)
    {
        return new GlobalAveragePooling1D(
            $this->backend,
            $options);
    }

    public function GlobalAveragePooling2D(
        array $options=null)
    {
        return new GlobalAveragePooling2D(
            $this->backend,
            $options);
    }

    public function GlobalAveragePooling3D(
        array $options=null)
    {
        return new GlobalAveragePooling3D(
            $this->backend,
            $options);
    }

    public function Dropout(float $rate,array $options=null)
    {
        return new Dropout($this->backend,$rate,$options);
    }

    public function BatchNormalization(array $options=null)
    {
        return new BatchNormalization($this->backend,$options);
    }

    public function Embedding(int $inputDim,int $outputDim, array $options=null)
    {
        return new Embedding($this->backend, $inputDim, $outputDim, $options);
    }

    public function SimpleRNN(int $units, array $options=null)
    {
        return new SimpleRNN($this->backend, $units, $options);
    }

    public function LSTM(int $units, array $options=null)
    {
        return new LSTM($this->backend, $units, $options);
    }

    public function GRU(int $units, array $options=null)
    {
        return new GRU($this->backend, $units, $options);
    }

    public function Attention(array $options=null)
    {
        return new Attention($this->backend, $options);
    }
}
