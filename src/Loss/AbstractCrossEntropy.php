<?php
namespace Rindow\NeuralNetworks\Loss;

use Interop\Polite\Math\Matrix\NDArray;
//use Rindow\NeuralNetworks\Activation\Activation;
use Rindow\NeuralNetworks\Gradient\Core\Variable;
use Rindow\NeuralNetworks\Gradient\Core\GradientTape;
use Rindow\NeuralNetworks\Support\GenericUtils;
use InvalidArgumentException;
use DomainException;

abstract class AbstractCrossEntropy extends AbstractGradient implements Loss//,Activation
{
    use GenericUtils;
    protected $backend;
    protected $outputs;
    protected $trues;
    protected $fromLogits = false;

    abstract protected function activationFunction(NDArray $inputs) : NDArray;
    abstract protected function diffActivationFunction(NDArray $dOutputs, NDArray $outputs) : NDArray;
    abstract protected function lossFunction(NDArray $trues, NDArray $predicts, bool $fromLogits) : float;
    abstract protected function diffLossFunction(NDArray $trues, NDArray $predicts, bool $fromLogits) : NDArray;

    public function __construct($backend,array $options=null)
    {
        extract($this->extractArgs([
            'from_logits' => false,
        ],$options));
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
/*
    public function forward(NDArray $inputs, bool $training) : NDArray
    {
        $K = $this->backend;
        $this->outputs = $this->activationFunction($inputs);
        return $this->outputs;
    }

    public function backward(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        if($this->fromLogits) {
            return $dOutputs;
        }
        return $this->diffActivationFunction($dOutputs, $this->outputs);
    }
*/
    //public function loss(NDArray $trues, NDArray $predicts) : float
    public function forward(NDArray $trues, NDArray $predicts) : float
    {
        $this->trues = $trues;
        if($this->fromLogits) {
            $predicts = $this->activationFunction($predicts);
        }
        $this->outputs = $predicts;
        return $this->lossFunction(
            $this->trues, $predicts, $this->fromLogits);
    }

    //public function differentiateLoss() : NDArray
    public function backward(array $dOutputs) : array
    {
        $dInputs = $this->diffLossFunction(
            $this->trues, $this->outputs, $this->fromLogits);
        return [$dInputs];
    }
}
