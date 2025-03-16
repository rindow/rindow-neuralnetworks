<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class AbstractGlobalAveragePooling extends AbstractImage
{
    use GenericUtils;
    protected int $reduceShape;
    protected string $pool_mode = 'avg';
    protected bool $channels_first = false;
    protected string $defaultLayerName;

    /**
     * @param array<int> $input_shape
     */
    public function __construct(
        object $backend,
        ?string $data_format=null,
        ?array $input_shape=null,
        ?string $name=null,
    )
    {
        // defaults
        $data_format = $data_format ?? null;
        $input_shape = $input_shape ?? null;

        parent::__construct($backend);
        $K = $backend;
        $this->data_format = $data_format;
        $this->inputShape = $input_shape;
        $this->initName($name,$this->defaultLayerName);
    }

    public function build(mixed $variable=null, ?array $sampleWeights=null) : void
    {
        $K = $this->backend;

        $inputShape = $this->normalizeInputShape($variable);
        //if(count($inputShape)!=1) {
        //    throw new InvalidArgumentException(
        ///        'Unsuppored input shape: ['.implode(',',$inputShape).']');
        //}
        $data_format = $this->data_format;
        if($data_format==null||
            $data_format=='channels_last') {
            $this->channels_first = false;
        } elseif($data_format=='channels_first') {
            $this->channels_first = true;
        } else {
            throw new InvalidArgumentException('data_format must be channels_last or channels_first');
        }
        if($this->channels_first) {
            $channels = array_unshift($inputShape);
        } else {
            $channels = array_pop($inputShape);
        }
        $this->reduceShape = (int)array_product($inputShape);
        $this->outputShape = [$channels];
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
                'data_format'=>$this->data_format,
            ]
        ];
    }

    protected function call(NDArray $inputs, ?bool $training=null) : NDArray
    {
        $K = $this->backend;
        $container = $this->container();
        $batches = $inputs->shape()[0];
        if($this->channels_first) {
            $inputsTmp = $inputs->reshape(
                array_merge([[$batches],$this->outputShape,[$this->reduceShape]]));
            $axis = -1;
        } else {
            $inputsTmp = $inputs->reshape(
                array_merge([$batches,$this->reduceShape],$this->outputShape));
            $axis = 1;
        }
        $outputs = $K->mean($inputsTmp, axis:$axis);
        $container->origInputsShape = $inputs->shape();
        return $outputs;
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        $container = $this->container();
        $dOutputs = $K->scale(1/$this->reduceShape,$dOutputs);
        if($this->channels_first) {
            $axis = -1; // [b,c] => [b,c,w]
        } else {
            $axis = 1;  // [b,c] => [b,w,c]
        }
        $dInputs = $K->repeat($dOutputs,$this->reduceShape,axis:$axis);

        return $dInputs->reshape($container->origInputsShape);
    }
}
