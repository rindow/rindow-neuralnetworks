<?php
namespace Rindow\NeuralNetworks\Builder;

use Rindow\NeuralNetworks\Optimizer\SGD;
use Rindow\NeuralNetworks\Optimizer\Adam;

class Optimizers
{
    protected $backend;

    public function __construct($backend)
    {
        $this->backend = $backend;
    }

    public function SGD(array $options=null)
    {
        return new SGD($this->backend, $options);
    }

    public function Adam(array $options=null)
    {
        return new Adam($this->backend, $options);
    }
}
