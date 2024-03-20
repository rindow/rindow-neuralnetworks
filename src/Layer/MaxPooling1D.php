<?php
namespace Rindow\NeuralNetworks\Layer;

use Interop\Polite\Math\Matrix\NDArray;

class MaxPooling1D extends AbstractPooling
{
    protected $rank = 1;
    protected $pool_mode = 'max';
    protected $defaultLayerName = 'maxpooling1d';

    protected function call(NDArray $inputs, bool $training=null) : NDArray
    {
        $K = $this->backend;
        $container = $this->container();
        $container->status = new \stdClass();
        $outputs = $K->pool1d(
                $container->status,
                $inputs,
                $this->poolSize,
                $this->strides,
                $this->padding,
                $this->data_format,
                $this->dilation_rate,
                $this->pool_mode
        );
        return $outputs;
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        $container = $this->container();
        $dInputs = $K->dPool1d(
            $container->status,
            $dOutputs
        );
        $container->status = null;
        return $dInputs;
    }
}
