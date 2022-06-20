<?php
namespace Rindow\NeuralNetworks\Layer;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class Dropout extends AbstractLayer
{
    use GenericUtils;
    protected $backend;
    protected $rate;
    //protected $mask;

    public function __construct(
        object $backend,
        float $rate,
        string $name=null,
        )
    {
        $this->backend = $K = $backend;
        $this->rate = min(1.0,max(0.0,$rate));
        $this->initName($name,'dropout');
    }

    public function getConfig() : array
    {
        return [
            'rate' => $this->rate
        ];
    }

    protected function call(NDArray $inputs, bool $training) : NDArray
    {
        $K = $this->backend;
        $container = $this->container();
        if($training) {
            $container->mask = $K->greater($K->randomUniformVariables($inputs->shape(),0.0,1.0), $this->rate);
            $outputs = $K->mul($inputs,$container->mask);
            return $outputs;
        } else {
            $outputs = $K->scale((1.0 - $this->rate), $inputs);
            return $outputs;
        }
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        $container = $this->container();
        $dInputs = $K->mul($dOutputs, $container->mask);
        return $dInputs;
    }

}
