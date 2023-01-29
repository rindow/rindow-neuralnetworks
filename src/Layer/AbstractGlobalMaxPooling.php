<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class AbstractGlobalMaxPooling extends AbstractImage
{
    use GenericUtils;
    protected $backend;
    protected $data_format;
    protected $reduceShape;
    protected $pool_mode = 'max';
    protected $channels_first = false;
    protected $defaultLayerName;

    public function __construct(
        object $backend,
        string $data_format=null,
        array $input_shape=null,
        string $name=null,
    )
    {
        // defaults
        $data_format = $data_format ?? null;
        $input_shape = $input_shape ?? null;

        $this->backend = $K = $backend;
        $this->data_format = $data_format;
        $this->inputShape = $input_shape;
        $this->initName($name,$this->defaultLayerName);
    }

    public function build($variable=null, array $sampleWeights=null)
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

    protected function call(NDArray $inputs, bool $training) : NDArray
    {
        $K = $this->backend;
        $container = $this->container();
        $batches = $inputs->shape()[0];
        // channels_last:  shape == [batches, imageshape, channels]
        // channels_first: shape == [batches, channels, imageshape]
        if($this->channels_first) {
            $reshapedInputs = $inputs->reshape([$batches,$this->outputShape[0],$this->reduceShape]);
            $axis = -1;
        } else {
            $reshapedInputs = $inputs->reshape([$batches,$this->reduceShape,$this->outputShape[0]]);
            $axis = 1;
        }
        $outputs = $K->max($reshapedInputs, $axis);
        $container->reshapedInputs = $reshapedInputs;
        $container->origInputsShape = $inputs->shape();
        return $outputs;
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        $container = $this->container();
        // input.shape == [batches,outshape,channels]
        // d max
        //dx = dy * onehot(argMax(x))
        // argMax.shape == [batches*channels]
        // dOutputs.shape == [batches*channels]
        if($this->channels_first) {
            $axis=-1;
        } else {
            $axis=1;
        }
        $argMax = $K->argMax($container->reshapedInputs,$axis);
        $dInputs = $K->scatter(
            $argMax,
            $dOutputs,
            $this->reduceShape,
            $axis
        );
        // channels_last:  dInputs.shape == [batches, outshape, channels]
        // channels_first: dInputs.shape == [batches, channels, outshape]

        return $dInputs->reshape($container->origInputsShape);
    }
}
