<?php
namespace Rindow\NeuralNetworks\Layer;

use Interop\Polite\Math\Matrix\NDArray;

class AveragePooling3D extends AbstractPooling implements Layer
{
    protected $rank = 3;
    protected $pool_mode = 'avg';

    protected function call(NDArray $inputs, bool $training) : NDArray
    {
        $K = $this->backend;
        $this->status = new \stdClass();
        $outputs = $K->pool3d(
                $this->status,
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
        $dInputs = $K->dPool3d(
            $this->status,
            $dOutputs
        );
        $this->status = null;
        return $dInputs;
    }
}
