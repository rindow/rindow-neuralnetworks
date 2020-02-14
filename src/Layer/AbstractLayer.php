<?php
namespace Rindow\NeuralNetworks\Layer;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

/**
 *
 */
abstract class AbstractLayer
{
    abstract protected function call(NDArray $inputs, bool $training) : NDArray;
    abstract protected function differentiate(NDArray $dOutputs) : NDArray;

    protected $inputShape;
    protected $outputShape;

    public function build(array $inputShape=null, array $options=null)
    {
        if($inputShape!==null)
            $this->inputShape = $inputShape;
        $this->outputShape = $inputShape;
    }

    public function outputShape() : array
    {
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
            //'input_shape' => $this->inputShape,
            //'output_shape' => $this->outputShape,
        ];
    }

    public function setName(string $name) : void
    {
        $this->name = $name;
    }

    public function getName() : string
    {
        return $this->name;
    }

    protected function assertInputShape(NDArray $inputs)
    {
        if($this->inputShape===null) {
            throw new InvalidArgumentException('Uninitialized');
        }
        $shape = $inputs->shape();
        $batchNum = array_shift($shape);
        if($shape!=$this->inputShape) {
            $shape = $shape ? implode(',',$shape) : '';
            throw new InvalidArgumentException('unmatch input shape: ['.$shape.'], must be ['.implode(',',$this->inputShape).']');
        }
    }

    protected function assertOutputShape(NDArray $outputs)
    {
        $shape = $outputs->shape();
        $batchNum = array_shift($shape);
        if($shape!=$this->outputShape) {
            throw new InvalidArgumentException('unmatch output shape: ['.
                implode(',',$shape).'], must be ['.implode(',',$this->outputShape).']');
        }
    }

    final public function forward(NDArray $inputs, bool $training) : NDArray
    {
        $this->assertInputShape($inputs);

        $outputs = $this->call($inputs, $training);

        $this->assertOutputShape($outputs);
        return $outputs;
    }

    final public function backward(NDArray $dOutputs) : NDArray
    {
        $this->assertOutputShape($dOutputs);

        $dInputs = $this->differentiate($dOutputs);

        $this->assertInputShape($dInputs);
        return $dInputs;
    }

}
