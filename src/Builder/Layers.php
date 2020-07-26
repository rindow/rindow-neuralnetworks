<?php
namespace Rindow\NeuralNetworks\Builder;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Layer\ReLU;
use Rindow\NeuralNetworks\Layer\Sigmoid;
use Rindow\NeuralNetworks\Layer\Softmax;
use Rindow\NeuralNetworks\Layer\Dense;
use Rindow\NeuralNetworks\Layer\Flatten;
use Rindow\NeuralNetworks\Layer\Conv1D;
use Rindow\NeuralNetworks\Layer\Conv2D;
use Rindow\NeuralNetworks\Layer\Conv3D;
use Rindow\NeuralNetworks\Layer\MaxPooling1D;
use Rindow\NeuralNetworks\Layer\MaxPooling2D;
use Rindow\NeuralNetworks\Layer\MaxPooling3D;
use Rindow\NeuralNetworks\Layer\AveragePooling1D;
use Rindow\NeuralNetworks\Layer\AveragePooling2D;
use Rindow\NeuralNetworks\Layer\AveragePooling3D;
use Rindow\NeuralNetworks\Layer\Dropout;
use Rindow\NeuralNetworks\Layer\BatchNormalization;

class Layers
{
    protected $backend;

    public function __construct($backend)
    {
        $this->backend = $backend;
    }

    public function ReLU(array $options=null)
    {
        return new ReLU($this->backend,$options);
    }

    public function Sigmoid(array $options=null)
    {
        return new Sigmoid($this->backend,$options);
    }

    public function Softmax(array $options=null)
    {
        return new Softmax($this->backend,$options);
    }

    public function Dense(int $units, array $options=null)
    {
        return new Dense($this->backend, $units, $options);
    }

    public function Flatten(
        array $options=null)
    {
        return new Flatten($this->backend, $options);
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

    public function Dropout(float $rate,array $options=null)
    {
        return new Dropout($this->backend,$rate,$options);
    }

    public function BatchNormalization(array $options=null)
    {
        return new BatchNormalization($this->backend,$options);
    }
}
