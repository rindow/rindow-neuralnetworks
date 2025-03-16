<?php
namespace Rindow\NeuralNetworks\Gradient;

interface Module
{
    public function name() : ?string;
    
    /**
     * @return array<Module>
     */
    public function submodules() : array;

    public function isAwareOf(string $name) : bool;
    
    /**
     * @return array<Variable>
     */
    public function variables() : array;

    /**
     * @return array<Variable>
     */
    public function trainableVariables() : array;

    public function reverseSyncWeightVariables() : void;
    
    public function setShapeInspection(bool $enable) : void;
}