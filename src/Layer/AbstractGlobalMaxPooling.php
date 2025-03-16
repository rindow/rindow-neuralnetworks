<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class AbstractGlobalMaxPooling extends AbstractImage
{
    use GenericUtils;
    protected int $reduceShape;
    protected string $pool_mode = 'max';
    protected bool $channels_first = false;
    protected string $defaultLayerName = 'unknow';

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
        // channels_first: inputs == (batches, channels, imageshape)
        // channels_last:  inputs == (batches, imageshape, channels)
        // outputs == (batches, channels)
        if($this->channels_first) {
            $reshapedInputs = $inputs->reshape([$batches,$this->outputShape[0],$this->reduceShape]);
            $axis = -1;
        } else {
            $reshapedInputs = $inputs->reshape([$batches,$this->reduceShape,$this->outputShape[0]]);
            $axis = 1;
        }
        $outputs = $K->max($reshapedInputs, axis:$axis);
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

        // channels_first: inputs == (batches, channels, imageshape)
        // channels_last:  inputs == (batches, imageshape, channels)
        if($this->channels_first) {
            $axis=2;
        } else {
            $axis=1;
        }
        // argMax == (batches, channels)
        $argMax = $K->argMax($container->reshapedInputs,axis:$axis,dtype:NDArray::int32);

        //$dInputs = $K->scatter(
        //    $argMax,
        //    $dOutputs,
        //    $this->reduceShape,
        //    axis:$axis
        //);

        // scatterb version
        // dOutputs == (batches, channels)
        // channels_last:  inputs == (batches, imageshape, channels)
        // channels_first: inputs == (batches, channels, imageshape)
        $dInputs = $K->scatterb(
            $argMax,                    // indices
            $dOutputs,                  // updates
            $container->reshapedInputs->shape(),// shape
            axis:$axis,
            batchDims:$axis,
            detailDepth:$container->reshapedInputs->ndim(),
            indexDepth:$axis,
        );

        // scatterND version
        //if($this->channels_first) {
        //    $axis=1;
        //    $shape = $container->reshapedInputs->shape();
        //} else {
        //    $axis=2;
        //    [$batches,$imageshape,$channels] = $container->reshapedInputs->shape();
        //    $shape = [$batches,$channels,$imageshape];
        //}
        //$argMax = $K->expandDims($argMax,axis:-1);
        //// argMax(x):  (batchs,channels,1)
        //// dOutput(b): (batchs,channels)
        //// dInputs(a): (batchs,channels,imageshape)
        //// batch_dims: batchs+channels
        ////echo "===============================\n";
        ////echo "channels_first=".($this->channels_first?'true':'false')."\n";
        ////echo "origInputs=(".implode(',',$container->origInputsShape).")\n";
        ////echo "shape=(".implode(',',$shape).")\n";
        ////echo "argMax=(".implode(',',$argMax->shape()).")\n";
        ////echo "dOutputs=(".implode(',',$dOutputs->shape()).")\n";
        ////echo "batchDims:$axis\n";
        //$dInputs = $K->scatterND(
        //    $argMax,
        //    $dOutputs,
        //    $shape,
        //    batchDims:$axis
        //);
        //if(!$this->channels_first) {
        //    $dInputs = $K->transpose($dInputs,perm:[0,2,1]);
        //}
        //// channels_last:  dInputs.shape == (batches, outshape, channels)
        //// channels_first: dInputs.shape == (batches, channels, outshape)
        ////echo "dInputs=(".implode(',',$dInputs->shape()).")\n";

        return $dInputs->reshape($container->origInputsShape);
    }
}

// max(axis=1) 
// inputs = (4,3,2)
// [
//   [[1,0],[0,1],[0,0]],
//   [[1,0],[0,1],[0,0]],
//   [[1,0],[0,1],[0,0]],
//   [[1,0],[0,1],[0,0]],
// ]
// x = (4,3)
// outputs = (4,2)
// [
//   [4,5],
//   [4,5],
// ]

