<?php
namespace Rindow\NeuralNetworks\Layer;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Module;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Activation\Activation;

/**
 *
 */
interface Layer extends Module
{
    /**
     * @param array<NDArray> $sampleWeights
     */
    public function build(mixed $variable=null,?array $sampleWeights=null) : void;

    public function isBuilt() : bool;

    /**
     * @return array<int|array<int>>
     */
    public function inputShape() : array;

    /**
     * @return array<int>
     */
    public function outputShape() : array;

    /**
     * @return array<NDArray>
     */
    public function getParams() : array;

    /**
     * @return array<NDArray>
     */
    public function getGrads() : array;

    public function setName(string $name) : void;

    public function name() : string;

    /**
    *  @return array<Variable>
    */
    public function weights() : array;

    /**
     * @return array<string,mixed>
     */
    public function getConfig() : array;

    public function getActivation() : ?Activation;
}
