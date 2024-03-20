<?php
namespace Rindow\NeuralNetworks\Layer;

use Interop\Polite\Math\Matrix\NDArray;

class Conv2D extends AbstractConv
{
    protected $rank = 2;
    protected $defaultLayerName = 'conv2d';

    protected function call(NDArray $inputs, bool $training=null) : NDArray
    {
        $K = $this->backend;
        $container = $this->container();
        $container->status = new \stdClass();
        $outputs = $K->conv2d(
                $container->status,
                $inputs,
                $this->kernel,
                $this->bias,
                $this->strides,
                $this->padding,
                $this->data_format,
                $this->dilation_rate
        );
        if($this->activation) {
            $container->activation = new \stdClass();
            $outputs = $this->activation->forward($container->activation,$outputs,training:$training);
        }
        return $outputs;
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        $container = $this->container();
        if($this->activation) {
            $dOutputs = $this->activation->backward($container->activation,$dOutputs);
        }
        $dInputs = $K->dConv2d(
            $container->status,
            $dOutputs,
            $this->dKernel,
            $this->dBias
        );
        $container->status = null;
        return $dInputs;
    }
}
