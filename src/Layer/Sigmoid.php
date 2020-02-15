<?php
namespace Rindow\NeuralNetworks\Layer;

use Interop\Polite\Math\Matrix\NDArray;

class Sigmoid extends AbstractLayer implements Layer
{
    protected $backend;
    protected $outputs;
    protected $incorporatedLoss = false;

    public function __construct($backend,array $options=null)
    {
        //extract($this->extractArgs([
        //],$options));
        $this->backend = $K = $backend;
    }

    protected function call(NDArray $inputs, bool $training) : NDArray
    {
        $K = $this->backend;
        $this->outputs = $K->sigmoid($inputs);
        return $this->outputs;
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        $dx = $K->onesLike($this->outputs);
        $K->update_sub($dx,$this->outputs);
        $K->update_mul($dx,$this->outputs);
        $K->update_mul($dx,$dOutputs);
        $dInputs = $dx;
        return $dInputs;
    }
}
