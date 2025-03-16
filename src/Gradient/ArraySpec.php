<?php
namespace Rindow\NeuralNetworks\Gradient;

interface ArraySpec
{
    public function shape() : ArrayShape;
    public function dtype() : int;
    public function name() : ?string;
}
