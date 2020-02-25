<?php
namespace Rindow\NeuralNetworks\Layer;

use Interop\Polite\Math\Matrix\NDArray;

class ReLU extends AbstractLayer implements Layer
{
    protected $backend;
    protected $mask;

    public function __construct($backend,array $options=null)
    {
        //extract($this->extractArgs([
        //],$options));
        $this->backend = $K = $backend;
    }

    protected function call(NDArray $inputs, bool $training) : NDArray
    {
        $K = $this->backend;
        $this->inputs = $inputs;
        $outputs = $K->relu($inputs);
        return $outputs;
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        //$mask = $K->cast($K->greater($this->inputs, 0.0),NDArray::float32);
        $mask = $K->greater($this->inputs,0.0);
        $dInputs = $K->mul($dOutputs,$mask);
        return $dInputs;
    }
}
