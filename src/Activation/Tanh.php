<?php
namespace Rindow\NeuralNetworks\Activation;

use Interop\Polite\Math\Matrix\NDArray;

class Tanh extends AbstractActivation
{
    protected function call(NDArray $inputs, ?bool $training=null) : NDArray
    {
        $K = $this->backend;
        $outputs = $K->tanh($inputs);
        $this->states->outputs = $outputs;
        return $outputs;
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        // dx = dy * (1 - y**2)
        $dInputs = $K->mul($dOutputs,$K->increment(
            $K->square($this->states->outputs),1,-1));
        return $dInputs;
    }
}
