<?php
namespace Rindow\NeuralNetworks\Layer;

use Interop\Polite\Math\Matrix\NDArray;

class Conv1D extends AbstractConv
{
    protected $rank = 1;
    protected $defaultLayerName = 'conv1d';

    protected function call(NDArray $inputs, bool $training) : NDArray
    {
        $K = $this->backend;
        $container = $this->container();
        $container->status = new \stdClass();
        $outputs = $K->conv1d(
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
            $outputs = $this->activation->forward($container->activation,$outputs,$training);
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
        $dInputs = $K->dConv1d(
            $container->status,
            $dOutputs,
            $this->dKernel,
            $this->dBias
        );
        $container->status = null;
        return $dInputs;
    }
}
