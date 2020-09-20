<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class Input extends AbstractLayer implements Layer
{
    use GenericUtils;
    protected $backend;

    public function __construct($backend,array $options=null)
    {
        extract($this->extractArgs([
            'shape'=>null,
        ],$options));
        $this->inputShape = $shape;
    }

    public function build(array $inputShape=null, array $options=null) : array
    {
        $K = $this->backend;

        $inputShape = $this->normalizeInputShape($inputShape);
        $outputShape = $inputShape;
        $this->outputShape = $outputShape;
        return $this->outputShape;
    }

    public function getParams() : array
    {
        return [];
    }

    public function getGrads() : array
    {
        return [];
    }

    public function getConfig() : array
    {
        return [
            'options' => [
                'input_shape'=>$this->inputShape,
            ]
        ];
    }

    protected function call(NDArray $inputs, bool $training) : NDArray
    {
        $K = $this->backend;
        return $inputs;
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        return $dOutputs;
    }
}
