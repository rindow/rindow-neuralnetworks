<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

abstract class AbstractPooling extends AbstractImage
{
    abstract protected function call(NDArray $inputs, bool $training=null) : NDArray;
    abstract protected function differentiate(NDArray $dOutputs) : NDArray;

    use GenericUtils;
    /** @var array<int> $poolSize */
    protected array $poolSize;
    /** @var array<int> $strides */
    protected array $strides;
    protected string $padding;
    /** @var array<int> $dilation_rate */
    protected array $dilation_rate;
    protected string $defaultLayerName = 'unkown';

    //protected $status;

    /**
     * @param int|array<int> $pool_size
     * @param int|array<int> $strides
     * @param int|array<int> $dilation_rate
     * @param array<int> $input_shape
     */
    public function __construct(
        object $backend,
        int|array $pool_size=null,
        int|array $strides=null,
        string $padding=null,
        string $data_format=null,
        int|array $dilation_rate=null,
        array $input_shape=null,
        string $name=null,
    )
    {
        // defaults
        $pool_size = $pool_size ?? 2;
        $strides = $strides ?? null;
        $padding = $padding ?? "valid";
        $data_format = $data_format ?? null;
        $dilation_rate = $dilation_rate ?? 1;
        $input_shape = $input_shape ?? null;

        parent::__construct($backend);
        $this->initName($name,$this->defaultLayerName);
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

    public function build(mixed $variable=null, array $sampleWeights=null) : void
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
