<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

abstract class AbstractConv extends AbstractImage
{
    abstract protected function call(NDArray $inputs, ?bool $training=null) : NDArray;
    abstract protected function differentiate(NDArray $dOutputs) : NDArray;

    use GenericUtils;
    protected int $filters;
    /** @var array<int> $kernel_size */
    protected array $kernel_size;
    /** @var array<int> $strides */
    protected array $strides;
    protected string $padding;
    /** @var array<int> $dilation_rate */
    protected array $dilation_rate;
    //protected $activation;
    protected bool $useBias;
    protected mixed $kernelInitializer;
    protected mixed $biasInitializer;
    protected ?string $kernelInitializerName;
    protected ?string $biasInitializerName;
    protected string $defaultLayerName;

    protected ?NDArray $kernel=null;
    protected NDArray $bias;
    protected NDArray $dKernel;
    protected NDArray $dBias;
    //protected $status;


    /**
     * @param int|array<int> $kernel_size
     * @param int|array<int> $strides
     * @param int|array<int> $dilation_rate
     * @param array<int> $input_shape
     */
    public function __construct(
        object $backend,
        int $filters,
        int|array $kernel_size,
        int|array|null $strides=null,
        ?string $padding=null,
        ?string $data_format=null,
        int|array|null $dilation_rate=null,
        ?int $groups=null,
        string|object|null $activation=null,
        ?bool $use_bias=null,
        string|callable|null $kernel_initializer=null,
        string|callable|null $bias_initializer=null,
        ?string $kernel_regularizer=null,
        ?string $bias_regularizer=null,
        ?string $activity_regularizer=null,
        ?string $kernel_constraint=null,
        ?string $bias_constraint=null,
        ?array $input_shape=null,
        ?string $name=null,
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

        parent::__construct($backend);
        $K = $backend;
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
        //$this->activation = $activation;
        $this->kernelInitializer = $K->getInitializer($kernel_initializer);
        $this->biasInitializer   = $K->getInitializer($bias_initializer);
        $this->kernelInitializerName = $kernel_initializer;
        $this->biasInitializerName = $bias_initializer;
        $this->useBias = $use_bias;
        $this->allocateWeights($this->useBias?['kernel','bias']:['kernel']);
        $this->setActivation($activation);
    }

    public function build(mixed $variable=null, ?array $sampleWeights=null) : void
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
        if($this->useBias) {
            return [$this->kernel,$this->bias];
        } else {
            return [$this->kernel];
        }
    }

    public function getGrads() : array
    {
        if($this->useBias) {
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
