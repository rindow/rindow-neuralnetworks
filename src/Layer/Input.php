<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class Input extends AbstractLayer
{
    use GenericUtils;
    protected $backend;

    public function __construct(
        object $backend,
        array $shape=null,
        string $name=null,
    )
    {
        $shape = $shape ?? null;
        $name = $name ?? null;
        
        $this->backend = $backend;
        $this->inputShape = $shape;
        $this->initName($name,'input');
    }

    public function build($variable=null, array $sampleWeights=null)
    {
        $K = $this->backend;

        $inputShape = $this->normalizeInputShape($variable);
        $outputShape = $inputShape;
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
            'options' => [
                'input_shape'=>$this->inputShape,
            ]
        ];
    }

    protected function call(NDArray $inputs, bool $training) : NDArray
    {
        $K = $this->backend;
        return $inputs;
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        return $dOutputs;
    }
}
