<?php
namespace Rindow\NeuralNetworks\Layer;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

/**
 *
 */
abstract class AbstractRNNCell extends AbstractLayerBase implements RNNCell
{
    abstract protected function call(NDArray $inputs, array $states, bool $training=null, object $calcState=null) : array;
    abstract protected function differentiate(NDArray $dOutputs, array $dStates, object $calcState) : array;

    public function getParams() : array
    {
        if($this->useBias) {
            return [$this->kernel,$this->recurrentKernel,$this->bias];
        } else {
            return [$this->kernel,$this->recurrentKernel];
        }
    }

    public function getGrads() : array
    {
        if($this->useBias) {
            return [$this->dKernel,$this->dRecurrentKernel,$this->dBias];
        } else {
            return [$this->dKernel,$this->dRecurrentKernel];
        }
    }

    public function reverseSyncCellWeightVariables(array $weights) : void
    {
        if($this->useBias) {
            $this->kernel = $weights[0]->value();
            $this->recurrentKernel = $weights[1]->value();
            $this->bias = $weights[2]->value();
        } else {
            $this->kernel = $weights[0]->value();
            $this->recurrentKernel = $weights[1]->value();
        }
    }

    final public function forward(NDArray $inputs, array $states, bool $training=null, object $calcState=null) : array
    {
        $this->assertInputShape($inputs,'forward');

        [$outputs,$states] = $this->call($inputs,$states,$training,$calcState);

        $this->assertOutputShape($outputs,'forward');
        return [$outputs,$states];
    }

    final public function backward(NDArray $dOutputs, array $dStates, object $calcState) : array
    {
        $this->assertOutputShape($dOutputs,'backward');

        [$dInputs,$dStates] = $this->differentiate($dOutputs,$dStates,$calcState);

        $this->assertInputShape($dInputs,'backward');
        return [$dInputs,$dStates];
    }

    public function __clone()
    {
        if(isset($this->kernel)) {
            $this->kernel = clone $this->kernel;
        }
        if(isset($this->recurrentKernel)) {
            $this->recurrentKernel = clone $this->recurrentKernel;
        }
        if(isset($this->bias)) {
            $this->bias = clone $this->bias;
        }
        if(isset($this->dKernel)) {
            $this->dKernel = clone $this->dKernel;
        }
        if(isset($this->dRecurrentKernel)) {
            $this->dRecurrentKernel = clone $this->dRecurrentKernel;
        }
        if(isset($this->dBias)) {
            $this->dBias = clone $this->dBias;
        }
    }
}
