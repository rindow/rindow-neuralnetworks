<?php
namespace Rindow\NeuralNetworks\Layer;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use Rindow\NeuralNetworks\Activation\Activation as ActivationInterface;

/**
 *
 */
abstract class AbstractRNNCell extends AbstractLayerBase implements RNNCell
{
    /**
     * @param array<NDArray> $states
     * @return array<NDArray>
     */
    abstract protected function call(
        NDArray $inputs,
        array $states,
        ?bool $training=null,
        ?object $calcState=null
    ) : array;

    /**
     * @param array<NDArray> $dStates
     * @return array{NDArray,array<NDArray>}
     */
    abstract protected function differentiate(
        array $dStates,
        object $calcState
    ) : array;

    protected bool $useBias;
    protected ?NDArray $kernel=null;
    protected ?NDArray $recurrentKernel=null;
    protected NDArray $bias;
    protected NDArray $dKernel;
    protected NDArray $dRecurrentKernel;
    protected ?NDArray $dBias=null;


    protected ?ActivationInterface $recurrentActivation;
    protected ?string $recurrentActivationName;
    protected mixed $kernelInitializer;
    protected mixed $recurrentInitializer;
    protected mixed $biasInitializer;
    protected string $kernelInitializerName;
    protected string $recurrentInitializerName;
    protected string $biasInitializerName;

    protected function setRecurrentActivation(null|string|ActivationInterface $activation) : void
    {
        $this->recurrentActivation = $this->createFunction($activation);
        $this->recurrentActivationName = $this->toStringName($activation);
    }

    protected function setKernelInitializer(
        mixed $kernelInitializer=null,
        mixed $recurrentInitializer=null,
        mixed $biasInitializer=null,
    ) : void
    {
        $K = $this->backend;
        $this->kernelInitializer = $K->getInitializer($kernelInitializer);
        $this->recurrentInitializer = $K->getInitializer($recurrentInitializer);
        $this->biasInitializer   = $K->getInitializer($biasInitializer);
        $this->kernelInitializerName = $this->toStringName($kernelInitializer);
        $this->recurrentInitializerName = $this->toStringName($recurrentInitializer);
        $this->biasInitializerName = $this->toStringName($biasInitializer);
    }

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

    final public function forward(
        NDArray $inputs,
        array $states,
        ?bool $training=null,
        ?object $calcState=null,
        ) : array
    {
        $this->assertInputShape($inputs,'forward');

        $states = $this->call($inputs,$states,$training,$calcState);

        $this->assertOutputShape($states[0],'forward');
        return $states;
    }

    final public function backward(
        array $dStates,
        object $calcState,
        ) : array
    {
        $this->assertOutputShape($dStates[0],'backward');

        [$dInputs,$dStates] = $this->differentiate($dStates,$calcState);

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
