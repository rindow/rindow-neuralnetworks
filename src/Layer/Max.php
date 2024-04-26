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
        int $axis=null,
        array $input_shape=null,
        string $name=null,
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

    public function build(mixed $variable=null, array $sampleWeights=null) : void
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

    protected function call(NDArray $inputs, bool $training=null) : NDArray
    {
        $K = $this->backend;
        $container = $this->container();
        $outputs = $K->max($inputs,axis:$this->realAxis);
        $container->inputs = $inputs;
        return $outputs;
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        $container = $this->container();
        $argMax = $K->argMax($container->inputs,axis:$this->realAxis);
        $dInputs = $K->scatter(
            $argMax,
            $dOutputs,
            $this->reduceNumClass,
            $this->realAxis
        );
        return $dInputs;
    }
}
