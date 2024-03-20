<?php
namespace Rindow\NeuralNetworks\Activation;

use Interop\Polite\Math\Matrix\NDArray;

class Softmax extends AbstractActivation
{
    protected function call(NDArray $inputs, bool $training=null) : NDArray
    {
        $K = $this->backend;
        $outputs = $K->softmax($inputs);
        $this->states->outputs = $outputs;
        return $outputs;
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        return $K->dSoftmax($dOutputs, $this->states->outputs);
    }
}
