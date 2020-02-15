<?php
namespace Rindow\NeuralNetworks\Layer;

use Interop\Polite\Math\Matrix\NDArray;

class Dropout extends AbstractLayer implements Layer
{
    protected $backend;
    protected $rate;
    protected $mask;

    public function __construct($backend,float $rate,array $options=null)
    {
        //extract($this->extractArgs([
        //],$options));
        $this->backend = $K = $backend;
        $this->rate = min(1.0,max(0.0,$rate));
    }

    public function getConfig() : array
    {
        return array_merge(parent::getConfig(),[
            'rate' => $this->rate,
        ]);
    }

    protected function call(NDArray $inputs, bool $training) : NDArray
    {
        $K = $this->backend;
        if($training) {
            $this->mask = $K->greater($K->rand($inputs->shape()), $this->rate);
            $outputs = $K->mul($inputs,$this->mask);
            return $outputs;
        } else {
            $outputs = $K->scale((1.0 - $this->rate), $inputs);
            return $outputs;
        }
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        $dInputs = $K->mul($dOutputs, $this->mask);
        return $dInputs;
    }

}
