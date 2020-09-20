<?php
namespace Rindow\NeuralNetworks\Activation;

use Interop\Polite\Math\Matrix\NDArray;

abstract class AbstractActivation implements Activation
{
    abstract protected function call(NDArray $inputs, bool $training) : NDArray;
    abstract protected function differentiate(NDArray $dOutputs) : NDArray;

    protected $states;
    public function __construct($backend)
    {
        $this->backend = $backend;
    }

    public function getStates() : object
    {
        return $this->states;
    }

    public function setStates($states) : void
    {
        $this->states = $states;
    }

    public function forward(NDArray $inputs, bool $training) : NDArray
    {
        $this->states = new \stdClass();
        return $this->call($inputs,$training);
    }

    public function backward(NDArray $dOutputs) : NDArray
    {
        return $this->differentiate($dOutputs);
    }
}
