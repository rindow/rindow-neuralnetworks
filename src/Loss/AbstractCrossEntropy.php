<?php
namespace Rindow\NeuralNetworks\Loss;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\Variable;
use Rindow\NeuralNetworks\Gradient\Core\GradientTape;
use InvalidArgumentException;
use DomainException;
use ArrayAccess;

abstract class AbstractCrossEntropy extends AbstractLoss implements Loss//,Activation
{
    protected $backend;
    protected $fromLogits = false;
    //protected $outputs;
    //protected $trues;

    abstract protected function activationFunction(NDArray $inputs) : NDArray;
    abstract protected function diffActivationFunction(NDArray $dOutputs, NDArray $outputs) : NDArray;
    abstract protected function lossFunction(NDArray $trues, NDArray $predicts, bool $fromLogits) : NDArray;
    abstract protected function diffLossFunction(NDArray $trues, NDArray $predicts, bool $fromLogits) : NDArray;

    public function __construct(
        object $backend,
        bool $from_logits=null
        )
    {
        // defaults
        $from_logits = $from_logits ?? false;

        $this->backend = $backend;
        $this->fromLogits = $from_logits;
    }

    public function setFromLogits(bool $fromLogits)
    {
        $this->fromLogits = $fromLogits;
    }

    public function fromLogits()
    {
        return $this->fromLogits;
    }

    public function getConfig() : array
    {
        return [
        ];
    }

    protected function call(NDArray $trues, NDArray $predicts) : NDArray
    {
        $container = $this->container();
        $container->trues = $trues;
        if($this->fromLogits) {
            $predicts = $this->activationFunction($predicts);
        }
        $container->outputs = $predicts;
        return $this->lossFunction(
            $container->trues, $predicts, $this->fromLogits);
    }

    protected function differentiate(array $dOutputs, ArrayAccess $grads=null, array $oidsToCollect=null) : array
    {
        $container = $this->container();
        $dInputs = $this->diffLossFunction(
            $container->trues, $container->outputs, $this->fromLogits);
        return [$dInputs];
    }
}
