<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

abstract class AbstractConv extends AbstractImage implements Layer
{
    abstract protected function call(NDArray $inputs, bool $training) : NDArray;
    abstract protected function differentiate(NDArray $dOutputs) : NDArray;

    use GenericUtils;
    protected $backend;
    protected $filters;
    protected $kernel_size;
    protected $strides;
    protected $padding;
    protected $data_format;
    protected $activation;
    protected $useBias;
    protected $kernelInitializer;
    protected $biasInitializer;

    protected $kernel;
    protected $bias;
    protected $dKernel;
    protected $dBias;
    protected $status;


    public function __construct($backend,int $filters, $kernel_size, array $options=null)
    {
        extract($this->extractArgs([
            'strides'=>1,
            'padding'=>"valid",
            'data_format'=>null,
            # 'dilation_rate'=>[1, 1],
            'groups'=>1,
            'activation'=>null,
            'use_bias'=>true,
            'kernel_initializer'=>"glorot_uniform",
            'bias_initializer'=>"zeros",
            'kernel_regularizer'=>null,
            'bias_regularizer'=>null,
            'activity_regularizer'=>null,
            'kernel_constraint'=>null,
            'bias_constraint'=>null,

            'input_shape'=>null,
            'activation'=>null,
            'use_bias'=>true,

        ],$options));
        $this->backend = $K = $backend;
        $kernel_size=$this->normalizeFilterSize($kernel_size,'kernel_size',
            null,true);
        $strides=$this->normalizeFilterSize($strides,'strides',1);
        $this->kernel_size = $kernel_size;
        $this->filters = $filters;
        $this->strides = $strides;
        $this->padding = $padding;
        $this->data_format = $data_format;
        $this->inputShape = $input_shape;
        $this->activation = $activation;
        $this->kernelInitializer = $K->getInitializer($kernel_initializer);
        $this->biasInitializer   = $K->getInitializer($bias_initializer);
        $this->kernelInitializerName = $kernel_initializer;
        $this->biasInitializerName = $bias_initializer;
        if($use_bias===null || $use_bias) {
            $this->useBias = true;
        }
        $this->setActivation($activation);
    }

    public function build(array $inputShape=null, array $options=null) : array
    {
        extract($this->extractArgs([
            'sampleWeights'=>null,
        ],$options));
        $K = $this->backend;
        $kernelInitializer = $this->kernelInitializer;
        $biasInitializer = $this->biasInitializer;

        $inputShape = $this->normalizeInputShape($inputShape);
        $kernel_size = $this->kernel_size;
        $outputShape =
            $K->calcConvOutputShape(
                $this->inputShape,
                $this->kernel_size,
                $this->strides,
                $this->padding,
                $this->data_format
            );
        array_push($outputShape,$this->filters);

        $channels = $this->getChannels();
        array_push($kernel_size,
            $channels);
        array_push($kernel_size,
            $this->filters);
        if($sampleWeights) {
            $this->kernel = $sampleWeights[0];
            $this->bias = $sampleWeights[1];
        } else {
            $tmpSize = array_product($this->kernel_size);
            $this->kernel = $kernelInitializer(
                $kernel_size,
                [$tmpSize*$channels,$tmpSize*$this->filters]
            );
            if($this->useBias) {
                $this->bias = $biasInitializer([$this->filters]);
            }
        }
        $this->dKernel = $K->zerosLike($this->kernel);
        if($this->useBias) {
            $this->dBias = $K->zerosLike($this->bias);
        }
        $this->outputShape = $outputShape;
        return $this->outputShape;
    }

    public function getParams() : array
    {
        if($this->bias) {
            return [$this->kernel,$this->bias];
        } else {
            return [$this->kernel];
        }
    }

    public function getGrads() : array
    {
        if($this->bias) {
            return [$this->dKernel,$this->dBias];
        } else {
            return [$this->dKernel];
        }
    }

    public function getConfig() : array
    {
        return [
            'filters' => $this->filters,
            'kernel_size' => $this->kernel_size,
            'options' => [
                'strides' => $this->strides,
                'padding' => $this->padding,
                'data_format' => $this->data_format,
                'input_shape'=>$this->inputShape,
                'kernel_initializer' => $this->kernelInitializerName,
                'bias_initializer' => $this->biasInitializerName,
                'activation'=>$this->activationName,
            ],
        ];
    }
}
