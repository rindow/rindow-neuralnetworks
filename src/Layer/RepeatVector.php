<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class RepeatVector extends AbstractLayer
{
    use GenericUtils;
    protected $backend;
    protected $repeats;

    public function __construct(
        object $backend,
        int $repeats,
        array $input_shape=null,
        string $name=null,
    )
    {
        $input_shape = $input_shape ?? null;
        $name = $name ?? null;
        
        $this->backend = $backend;
        $this->repeats = $repeats;
        $this->inputShape = $input_shape;
        $this->initName($name,'repeatvector');
    }

    public function build($variable=null, array $sampleWeights=null)
    {
        $K = $this->backend;
        $inputShape = $this->normalizeInputShape($variable);
        if(count($inputShape)!=1) {
            throw new InvalidArgumentException('input shape must be 1D:'.implode(',',$inputShape));
        }
        array_unshift($inputShape,$this->repeats);
        $this->outputShape = $inputShape;
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
            'repeats' => $this->repeats,
            'options' => [
                'input_shape'=>$this->inputShape,
            ]
        ];
    }

    protected function call(NDArray $inputs, bool $training) : NDArray
    {
        $K = $this->backend;
        $outputs = $K->repeat($inputs,$this->repeats,$axis=1);
        return $outputs;
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        $dInput = $K->sum($dOutputs,$axis=1);
        return $dInput;
    }
}
