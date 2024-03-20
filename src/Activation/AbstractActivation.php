<?php
namespace Rindow\NeuralNetworks\Activation;

use Interop\Polite\Math\Matrix\NDArray;

abstract class AbstractActivation implements Activation
{
    abstract protected function call(NDArray $inputs, bool $training=null) : NDArray;
    abstract protected function differentiate(NDArray $dOutputs) : NDArray;

    protected $states;
    protected $backend;
    public function __construct(object $backend)
    {
        $this->backend = $backend;
    }

    //public function setStates($states) : void
    //{
    //    $this->states = $states;
    //}

    public function forward(object $states, NDArray $inputs, bool $training=null) : NDArray
    {
        //if($this->states===null) {
        //    $this->states = new \stdClass();
        //}
        $this->states = $states;
        try {
            $outputs = $this->call($inputs,$training);
        } finally {
            $this->states = null;
        }
        return $outputs;
    }

    public function backward(object $states, NDArray $dOutputs) : NDArray
    {
        $this->states = $states;
        try {
            $dInputs = $this->differentiate($dOutputs);
        } finally {
            $this->states = null;
        }
        return $dInputs;
    }
}
