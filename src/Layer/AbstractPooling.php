<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

abstract class AbstractPooling extends AbstractImage implements Layer
{
    abstract protected function call(NDArray $inputs, bool $training) : NDArray;
    abstract protected function differentiate(NDArray $dOutputs) : NDArray;

    use GenericUtils;
    protected $backend;
    protected $poolSize;
    protected $strides;
    protected $padding;
    protected $data_format;
    protected $dilation_rate;
    protected $status;

    public function __construct($backend,array $options=null)
    {
        extract($this->extractArgs([
            'pool_size'=>2,
            'strides'=>null,
            'padding'=>"valid",
            'data_format'=>null,
            'dilation_rate'=>1,
            'input_shape'=>null,
        ],$options));
        $this->backend = $backend;
        $pool_size=$this->normalizeFilterSize($pool_size,'pool_size',2);
        $strides=$this->normalizeFilterSize($strides,'strides',$pool_size);
        $dilation_rate=$this->normalizeFilterSize($dilation_rate,'dilation_rate',1);
        $this->poolSize = $pool_size;
        $this->strides = $strides;
        $this->padding = $padding;
        $this->data_format = $data_format;
        $this->dilation_rate = $dilation_rate;
        $this->inputShape = $input_shape;
    }

    public function build($variable=null, array $options=null)
    {
        $K = $this->backend;

        $inputShape = $this->normalizeInputShape($variable);
        $channels = $this->getChannels();
        $outputShape =
            $K->calcConvOutputShape(
                $this->inputShape,
                $this->poolSize,
                $this->strides,
                $this->padding,
                $this->data_format,
                $this->dilation_rate
            );
        array_push($outputShape,$channels);
        $this->outputShape = $outputShape;
        return $this->createOutputDefinition([$this->outputShape]);
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
                'pool_size' => $this->poolSize,
                'strides' => $this->strides,
                'padding' => $this->padding,
                'data_format' => $this->data_format,
                'input_shape'=>$this->inputShape,
            ]
        ];
    }
}
