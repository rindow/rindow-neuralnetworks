<?php
namespace Rindow\NeuralNetworks\Builder;

use Rindow\NeuralNetworks\Optimizer\SGD;
use Rindow\NeuralNetworks\Optimizer\Adam;
use Rindow\NeuralNetworks\Optimizer\RMSprop;

class Optimizers
{
    protected $backend;

    public function __construct(object $backend)
    {
        $this->backend = $backend;
    }

    public function SGD(...$options)
    {
        return new SGD($this->backend, ...$options);
    }

    public function Adam(...$options)
    {
        return new Adam($this->backend, ...$options);
    }

    public function RMSprop(...$options)
    {
        return new RMSprop($this->backend, ...$options);
    }
}
