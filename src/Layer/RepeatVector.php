<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class RepeatVector extends AbstractLayer implements Layer
{
    use GenericUtils;
    protected $backend;

    public function __construct(
        $backend,
        int $repeats,
        array $options=null)
    {
        extract($this->extractArgs([
            'input_shape'=>null,
        ],$options));
        $this->backend = $backend;
        $this->repeats = $repeats;
        $this->inputShape = $input_shape;
    }

    public function build(array $inputShape=null, array $options=null) : array
    {
        $K = $this->backend;
        $inputShape = $this->normalizeInputShape($inputShape);
        if(count($inputShape)!=1) {
            throw new InvalidArgumentException('input shape must be 1D:'.implode(',',$inputShape));
        }
        array_unshift($inputShape,$this->repeats);
        $this->outputShape = $inputShape;
        return $this->outputShape;
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
        $outputs = $K->repeat($inputs,$this->repeats);
        return $outputs;
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        $dInput = $K->reduceSumRepeated($dOutputs);
        return $dInput;
    }
}
