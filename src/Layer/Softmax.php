<?php
namespace Rindow\NeuralNetworks\Layer;

use Interop\Polite\Math\Matrix\NDArray;

class Softmax extends AbstractLayer implements Layer
{
    protected $backend;
    protected $outputs;

    public function __construct($backend,array $options=null)
    {
        //extract($this->extractArgs([
        //],$options));
        $this->backend = $K = $backend;
    }

    protected function call(NDArray $inputs, bool $training) : NDArray
    {
        $K = $this->backend;
        $this->outputs = $K->softmax($inputs);
        return $this->outputs;
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        return $K->dSoftmax($dOutputs, $this->outputs);
    }
}
