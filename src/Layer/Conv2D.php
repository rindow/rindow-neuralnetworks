<?php
namespace Rindow\NeuralNetworks\Layer;

use Interop\Polite\Math\Matrix\NDArray;

class Conv2D extends AbstractConv implements Layer
{
    protected $rank = 2;

    protected function call(NDArray $inputs, bool $training) : NDArray
    {
        $K = $this->backend;
        $this->status = new \stdClass();
        $outputs = $K->conv2d(
                $this->status,
                $inputs,
                $this->kernel,
                $this->bias,
                $this->strides,
                $this->padding,
                $this->data_format,
                $this->dilation_rate
        );
        if($this->activation)
            $outputs = $this->activation->forward($outputs,$training);
        return $outputs;
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        if($this->activation)
            $dOutputs = $this->activation->backward($dOutputs);
        $dInputs = $K->dConv2d(
            $this->status,
            $dOutputs,
            $this->dKernel,
            $this->dBias
        );
        $this->status = null;
        return $dInputs;
    }
}
