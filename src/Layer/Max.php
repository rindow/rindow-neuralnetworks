<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class Max extends AbstractLayer
{
    use GenericUtils;
    protected int $axis;
    protected int $realAxis;
    protected int $reduceNumClass;

    /**
     * @param array<int> $input_shape
     */
    public function __construct(
        object $backend,
        ?int $axis=null,
        ?array $input_shape=null,
        ?string $name=null,
    )
    {
        $axis = $axis ?? -1;
        $input_shape = $input_shape ?? null;
        $name = $name ?? null;
        
        parent::__construct($backend);
        $this->axis = $axis;
        $this->inputShape = $input_shape;
        $this->initName($name,'max');
    }

    public function build(mixed $variable=null, ?array $sampleWeights=null) : void
    {
        $K = $this->backend;

        $inputShape = $this->normalizeInputShape($variable);

        $axis = $this->axis;
        if($axis < 0) {
            $axis = count($inputShape) + $axis;
        }
        if($axis<0||$axis>count($inputShape)) {
            throw new InvalidArgumentException(
                'Invalid axis. Dims of the inputshape is '.count($inputShape).'. axis='.$this->axis.' given');
        }
        $right = $inputShape;
        $left = [];
        for($i=0;$i<$axis;$i++) {
            $left[] = array_shift($right);
        }
        $this->reduceNumClass = array_shift($right);
        $outputShape = array_merge($left,$right);

        $this->realAxis = $axis+1;

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
            'axis' => $this->axis,
            'options' => [
                'input_shape'=>$this->inputShape,
            ]
        ];
    }

    protected function call(NDArray $inputs, ?bool $training=null) : NDArray
    {
        $K = $this->backend;
        $container = $this->container();
        // inputs: (prefix,channels,postfix)
        // output: (prefix,postfix)
        $outputs = $K->max($inputs,axis:$this->realAxis);
        $container->inputs = $inputs;
        return $outputs;
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        $container = $this->container();
        // inputs: (prefix,channels,postfix)
        // argMax: (prefix,postfix)
        $argMax = $K->argMax($container->inputs,axis:$this->realAxis,dtype:NDArray::int32);

        //$dInputs = $K->scatter(
        //    $argMax,
        //    $dOutputs,
        //    $this->reduceNumClass,
        //    $this->realAxis
        //);
        $dInputs = $K->scatterb(
            $argMax,                    // indices
            $dOutputs,                  // updates
            $container->inputs->shape(),// shape
            axis:$this->realAxis,
            batchDims:$this->realAxis,
            detailDepth:$container->inputs->ndim(),
            indexDepth:$this->realAxis,
        );

        //$postfix = $container->inputs->shape();
        //$prefix = array_splice($postfix,0,$this->realAxis);
        //$channels = array_shift($postfix);
        //$shape = array_merge($prefix,$postfix,[$channels]);
        //$argMax = $K->expandDims($argMax,axis:-1);
        //// argMax:   (prefix,postfix,1)
        //// dOutputs: (prefix,postfix)
        //// dInputs:  (prefix,postfix,channels)
        //// batch_dims: prefix+postfix
        ////echo "===============================\n";
        ////echo "origInputs=(".implode(',',$container->inputs->shape()).")\n";
        ////echo "axis:".$this->realAxis."\n";
        ////echo "prefix=(".implode(',',$prefix).")\n";
        ////echo "channels:".$channels."\n";
        ////echo "postfix=(".implode(',',$postfix).")\n";
        ////echo "shape=(".implode(',',$shape).")\n";
        ////echo "argMax=(".implode(',',$argMax->shape()).")\n";
        ////echo "dOutputs=(".implode(',',$dOutputs->shape()).")\n";
        ////echo "batchDims:".($container->inputs->ndim()-1)."\n";
        //$dInputs = $K->scatterND(
        //    $argMax,
        //    $dOutputs,
        //    $shape,
        //    batchDims:$container->inputs->ndim()-1
        //);

        //// dInputs:  (prefix,postfix,channels)
        //$depth = $container->inputs->ndim();
        //$perm_prefix = [];
        //$perm_channels = [];
        //$perm_postfix = [];
        //if($this->realAxis > 0) {
        //    $perm_prefix = range(0,$this->realAxis-1);
        //}
        //if($depth-1>=0) {
        //    $perm_channels = [$depth-1];
        //}
        //if($depth>$this->realAxis+1) {
        //    $perm_postfix = range($this->realAxis, $depth-2);
        //}
        //$perm = array_merge($perm_prefix,$perm_channels,$perm_postfix);
        ////echo "depth:".$depth."\n";
        ////echo "realAxis:".$this->realAxis."\n";
        ////echo "perm_prefix=(".implode(',',$perm_prefix).")\n";
        ////echo "perm_channels=(".implode(',',$perm_channels).")\n";
        ////echo "perm_postfix=(".implode(',',$perm_postfix).")\n";
        ////echo "perm=(".implode(',',$perm).")\n";
        //$trans = false;
        //foreach($perm as $key=>$value) {
        //    if($key!=$value) {
        //        $trans = true;
        //    }
        //}
        //if($trans) {
        //    $dInputs = $K->transpose($dInputs,perm:$perm);
        //}
        ////echo "dInputs=(".implode(',',$dInputs->shape()).")\n";
        return $dInputs;
    }
}
