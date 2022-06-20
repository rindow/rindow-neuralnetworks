<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class Gather extends AbstractMultiInputLayer
{
    use GenericUtils;
    protected $backend;
    protected $axis;

    public function __construct($backend,array $options=null)
    {
        extract($this->extractArgs([
            'axis'=>-1,
            'input_shapes'=>null,
        ],$options));
        $this->backend = $backend;
        $this->axis = $axis;
        $this->inputShape = $input_shapes;
    }

    public function build($variables=null, array $options=null)
    {
        $K = $this->backend;

        $inputShapes = $this->normalizeInputShape($variables);
        if(count($inputShapes)!=2) {
            throw new InvalidArgumentException('num of inputs must be 2: inputs is '.count($inputShapes));
        }
        foreach ($inputShapes as $idx => $shape) {
            if(!is_array($shape)) {
                $type = gettype($shape);
                throw new InvalidArgumentException('input_shapes must be the list of shape: '.$type.' included in #'.$idx.'.');
            }
        }
        [$sourceShape,$indexShape] = $inputShapes;
        if($this->axis===null) {
            throw new InvalidArgumentException('Null axis is not supported.');
        }
        $axis = $this->axis;
        if($axis < 0) {
            $axis = count($sourceShape) + $axis;
        }
        if($axis<0||$axis>count($sourceShape)) {
            throw new InvalidArgumentException(
                'Invalid axis. Dims of the sourceShape is '.count($sourceShape).'. axis='.$this->axis.' given');
        }

        $postfixShape = $sourceShape;
        $prefixShape = [];
        for($i=0;$i<$axis;$i++) {
            $prefixShape[] = array_shift($postfixShape);
        }
        $this->reduceNumClass = array_shift($postfixShape);
        $outputShape = array_merge($prefixShape,$postfixShape);
        if($indexShape!=$outputShape) {
            throw new InvalidArgumentException('Unmatch source and index Shape and axis:'.
                    $this->shapeToString($sourceShape).','.
                    $this->shapeToString($outputShape).','.$this->axis);
        }

        $this->realAxis = $axis+1;

        $this->outputShape = $outputShape;
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
            'axis' => $this->axis,
            'options' => [
                'input_shape'=>$this->inputShape,
            ]
        ];
    }

    protected function call(array $inputs, bool $training) : NDArray
    {
        $K = $this->backend;
        [$source,$indexes] = $inputs;
        $outputs = $K->gather($source,$indexes,$this->realAxis);
        $this->indexes = $indexes;
        return $outputs;
    }

    protected function differentiate(NDArray $dOutputs) : array
    {
        $K = $this->backend;
        $dSource = $K->scatter(
            $this->indexes,
            $dOutputs,
            $this->reduceNumClass,
            $this->realAxis
        );
        $dIndex = $K->zerosLike($this->indexes);
        return [$dSource,$dIndex];
    }
}
