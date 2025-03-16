<?php
namespace Rindow\NeuralNetworks\Activation;

use Interop\Polite\Math\Matrix\NDArray;

class ReLU extends AbstractActivation
{
    #protected $mask;

    protected function call(NDArray $inputs, ?bool $training=null) : NDArray
    {
        $K = $this->backend;
        $this->states->inputs = $inputs;
        $outputs = $K->relu($inputs);
        return $outputs;
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        //$mask = $K->cast($K->greater($this->inputs, 0.0),NDArray::float32);
        $mask = $K->greater($this->states->inputs,0.0);
        $dInputs = $K->mul($dOutputs,$mask);
        return $dInputs;
    }
}
