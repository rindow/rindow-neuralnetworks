<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class Flatten extends AbstractLayer implements Layer
{
    use GenericUtils;
    protected $backend;

    public function __construct($backend,array $options=null)
    {
        extract($this->extractArgs([
            'input_shape'=>null,
        ],$options));
        $this->inputShape = $input_shape;
    }

    public function build($variable=null, array $options=null)
    {
        $K = $this->backend;

        $inputShape = $this->normalizeInputShape($variable);
        $outputShape = (int)array_product($inputShape);
        $this->outputShape = [$outputShape];
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
            ]
        ];
    }

    protected function call(NDArray $inputs, bool $training) : NDArray
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
