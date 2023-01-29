<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

abstract class AbstractConv extends AbstractImage
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
    protected $dilation_rate;
    protected $activation;
    protected $useBias;
    protected $kernelInitializer;
    protected $biasInitializer;
    protected $kernelInitializerName;
    protected $biasInitializerName;
    protected $defaultLayerName;

    protected $kernel;
    protected $bias;
    protected $dKernel;
    protected $dBias;
    //protected $status;


    public function __construct(
        object $backend,
        int $filters,
        int|array $kernel_size,
        int|array $strides=null,
        string $padding=null,
        string $data_format=null,
        int|array $dilation_rate=null,
        int $groups=null,
        string|object $activation=null,
        bool $use_bias=null,
        string|callable $kernel_initializer=null,
        string|callable $bias_initializer=null,
        string $kernel_regularizer=null,
        string $bias_regularizer=null,
        string $activity_regularizer=null,
        string $kernel_constraint=null,
        string $bias_constraint=null,
        array $input_shape=null,
        string $name=null,
    )
    {
        // defaults 
        $strides = $strides ?? 1;
        $padding = $padding ?? "valid";
        $data_format = $data_format ?? null;
        $dilation_rate = $dilation_rate ?? 1;
        $groups = $groups ?? 1;
        $activation = $activation ?? null;
        $use_bias = $use_bias ?? true;
        $kernel_initializer = $kernel_initializer ?? "glorot_uniform";
        $bias_initializer = $bias_initializer ?? "zeros";
        $kernel_regularizer = $kernel_regularizer ?? null;
        $bias_regularizer = $bias_regularizer ?? null;
        $activity_regularizer = $activity_regularizer ?? null;
        $kernel_constraint = $kernel_constraint ?? null;
        $bias_constraint = $bias_constraint ?? null;
        $input_shape = $input_shape ?? null;


        $this->backend = $K = $backend;
        $this->initName($name,$this->defaultLayerName);
        $kernel_size=$this->normalizeFilterSize($kernel_size,'kernel_size',
            null,true);
        $strides=$this->normalizeFilterSize($strides,'strides',1);
        $dilation_rate=$this->normalizeFilterSize($dilation_rate,'dilation_rate',1);
        $this->kernel_size = $kernel_size;
        $this->filters = $filters;
        $this->strides = $strides;
        $this->padding = $padding;
        $this->data_format = $data_format;
        $this->dilation_rate = $dilation_rate;
        $this->inputShape = $input_shape;
        $this->activation = $activation;
        $this->kernelInitializer = $K->getInitializer($kernel_initializer);
        $this->biasInitializer   = $K->getInitializer($bias_initializer);
        $this->kernelInitializerName = $kernel_initializer;
        $this->biasInitializerName = $bias_initializer;
        if($use_bias===null || $use_bias) {
            $this->useBias = true;
        }
        $this->allocateWeights($this->useBias?2:1);
        $this->setActivation($activation);
    }

    public function build($variable=null, array $sampleWeights=null)
    {
        $K = $this->backend;
        $kernelInitializer = $this->kernelInitializer;
        $biasInitializer = $this->biasInitializer;

        $inputShape = $this->normalizeInputShape($variable);
        $kernel_size = $this->kernel_size;
        $outputShape =
            $K->calcConvOutputShape(
                $this->inputShape,
                $this->kernel_size,
                $this->strides,
                $this->padding,
                $this->data_format,
                $this->dilation_rate
            );
        array_push($outputShape,$this->filters);

        $channels = $this->getChannels();
        array_push($kernel_size,$channels);
        array_push($kernel_size,$this->filters);
        if($this->kernel===null) {
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
        }
        $this->dKernel = $K->zerosLike($this->kernel);
        if($this->useBias) {
            $this->dBias = $K->zerosLike($this->bias);
        }
        $this->outputShape = $outputShape;
        $this->syncWeightVariables();
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

    public function reverseSyncWeightVariables() : void
    {
        if($this->useBias) {
            $this->kernel = $this->weights[0]->value();
            $this->bias = $this->weights[1]->value();
        } else {
            $this->kernel = $this->weights[0]->value();
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
