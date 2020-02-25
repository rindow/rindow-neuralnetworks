<?php
namespace Rindow\NeuralNetworks\Builder;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Layer\ReLU;
use Rindow\NeuralNetworks\Layer\Sigmoid;
use Rindow\NeuralNetworks\Layer\Softmax;
use Rindow\NeuralNetworks\Layer\Dropout;
use Rindow\NeuralNetworks\Layer\Dense;
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

    public function Dropout(float $rate,array $options=null)
    {
        return new Dropout($this->backend,$rate,$options);
    }

    public function BatchNormalization(array $options=null)
    {
        return new BatchNormalization($this->backend,$options);
    }
}
