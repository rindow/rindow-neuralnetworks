<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class AbstractGlobalAveragePooling extends AbstractImage implements Layer
{
    use GenericUtils;
    protected $backend;
    protected $pool_mode = 'avg';
    protected $channels_first = false;

    public function __construct($backend, array $options=null)
    {
        extract($this->extractArgs([
            'data_format'=>null,
            'input_shape'=>null,
        ],$options));
        $this->backend = $K = $backend;
        $this->data_format = $data_format;
        $this->inputShape = $input_shape;
    }

    public function build($variable=null, array $options=null)
    {
        extract($this->extractArgs([
            'sampleWeights'=>null,
        ],$options));
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
                'input_shape'=>$this->inputShape,
                'data_format'=>$this->data_format,
            ]
        ];
    }

    protected function call(NDArray $inputs, bool $training) : NDArray
    {
        $K = $this->backend;
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
        $outputs = $K->mean($inputsTmp, $axis);
        $this->origInputsShape = $inputs->shape();
        return $outputs;
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        $dOutputs = $K->scale(1/$this->reduceShape,$dOutputs);
        if($this->channels_first) {
            $axis = -1; // [b,c] => [b,c,w]
        } else {
            $axis = 1;  // [b,c] => [b,w,c]
        }
        $dInputs = $K->repeat($dOutputs,$this->reduceShape,$axis);

        return $dInputs->reshape($this->origInputsShape);
    }
}
