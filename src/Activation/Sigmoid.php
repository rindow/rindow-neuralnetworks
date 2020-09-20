<?php
namespace Rindow\NeuralNetworks\Activation;

use Interop\Polite\Math\Matrix\NDArray;

class Sigmoid extends AbstractActivation
{
    protected $incorporatedLoss = false;

    protected function call(NDArray $inputs, bool $training) : NDArray
    {
        $K = $this->backend;
        $outputs = $K->sigmoid($inputs);
        $this->states->outputs = $outputs;
        return $outputs;
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        $dInputs = $K->dSigmoid($dOutputs, $this->states->outputs);
        return $dInputs;
    }
}
