<?php
namespace Rindow\NeuralNetworks\Gradient;

interface Module
{
    public function submodules() : array;
    public function variables() : array;
    public function trainableVariables() : array;
}