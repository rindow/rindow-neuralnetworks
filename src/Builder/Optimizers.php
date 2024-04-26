<?php
namespace Rindow\NeuralNetworks\Builder;

use Rindow\NeuralNetworks\Optimizer\SGD;
use Rindow\NeuralNetworks\Optimizer\Adam;
use Rindow\NeuralNetworks\Optimizer\RMSprop;

class Optimizers
{
    protected object $backend;

    public function __construct(object $backend)
    {
        $this->backend = $backend;
    }

    public function SGD(mixed ...$options) : object
    {
        return new SGD($this->backend, ...$options);
    }

    public function Adam(mixed ...$options) : object
    {
        return new Adam($this->backend, ...$options);
    }

    public function RMSprop(mixed ...$options) : object
    {
        return new RMSprop($this->backend, ...$options);
    }
}
