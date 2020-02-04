<?php
namespace Rindow\NeuralNetworks\Layer;

use Interop\Polite\Math\Matrix\NDArray;

/**
 *
 */
interface Layer
{
    public function build(array $inputShape=null,array $options=null);
    public function outputShape() : array;
    public function getParams() : array;
    public function getGrads() : array;
    public function forward(NDArray $inputs, bool $training) : NDArray;
    public function backward(NDArray $dOutputs) : NDArray;
    public function setName(string $name) : void;
    public function getName() : string;
    public function getConfig() : array;
}
