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
    protected $realAxis;
    protected $reduceNumClass;

    public function __construct(
        object $backend,
        int $axis=null,
        array $input_shapes=null,
        string $name=null,
    )
    {
        // defaults
        $axis = $axis ?? -1;
        $input_shapes = $input_shapes ?? null;
        $name = $name ?? null;
        
        $this->backend = $backend;
        $this->axis = $axis;
        $this->inputShape = $input_shapes;
        $this->initName($name,'gather');
    }

    public function build($variables=null, array $sampleWeights=null)
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

    protected function call(array $inputs, bool $training=null) : NDArray
    {
        $K = $this->backend;
        $container = $this->container();
        [$source,$indexes] = $inputs;
        $outputs = $K->gather($source,$indexes,$this->realAxis);
        $container->indexes = $indexes;
        return $outputs;
    }

    protected function differentiate(NDArray $dOutputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        $dSource = $K->scatter(
            $container->indexes,
            $dOutputs,
            $this->reduceNumClass,
            $this->realAxis
        );
        $dIndex = $K->zerosLike($container->indexes);
        return [$dSource,$dIndex];
    }

}
