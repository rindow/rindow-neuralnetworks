<?php
namespace Rindow\NeuralNetworks\Layer;

use Interop\Polite\Math\Matrix\NDArray;

/**
 *
 */
interface LayerBase
{
    public function build($variable=null,array $options=null);
    public function outputShape() : array;
    public function getParams() : array;
    public function getGrads() : array;
    public function setName(string $name) : void;
    public function getName() : string;
    public function getConfig() : array;
    public function getActivation();
    public function setActivation($activation);
}
