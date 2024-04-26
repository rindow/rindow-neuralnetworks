<?php
namespace Rindow\NeuralNetworks\Support\HDA;

interface HDAFactory
{
    public function open(string|object $filename, string $mode=null) : HDA;
}
