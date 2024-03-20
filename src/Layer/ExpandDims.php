<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class ExpandDims extends AbstractLayer
{
    use GenericUtils;
    protected $backend;
    protected $axis;

    public function __construct(
        object $backend,
        int $axis,
        array $input_shape=null,
        string $name=null,
    )
    {
        $input_shape = $input_shape ?? null;
        $name = $name ?? null;
        
        $this->backend = $backend;
        $this->axis = $axis;
        $this->inputShape = $input_shape;
        $this->initName($name,'expanddims');
    }

    public function build($variable=null, array $sampleWeights=null)
    {
        $K = $this->backend;

        $inputShape = $this->normalizeInputShape($variable);

        $axis = $this->axis;
        if($axis < 0) {
            $axis = count($inputShape) + 1 + $axis;
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
        if($right===null) {
            $right=[];
        }
        $outputShape = array_merge($left,[1],$right);
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
            'dims' => $this->dims,
            'options' => [
                'input_shape'=>$this->inputShape,
            ]
        ];
    }

    protected function call(NDArray $inputs, bool $training=null) : NDArray
    {
        $K = $this->backend;
        $shape = $inputs->shape();
        $batch = array_shift($shape);
        $shape = $this->outputShape;
        array_unshift($shape,$batch);
        return $inputs->reshape($shape);
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        $shape = $dOutputs->shape();
        $batch = array_shift($shape);
        $shape = $this->inputShape;
        array_unshift($shape,$batch);
        return $dOutputs->reshape($shape);
    }
}
