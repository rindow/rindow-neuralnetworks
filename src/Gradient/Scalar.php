<?php
namespace Rindow\NeuralNetworks\Gradient;

interface Scalar
{
    public function value() : mixed;
    public function update(bool|int|float $value) : void;
}