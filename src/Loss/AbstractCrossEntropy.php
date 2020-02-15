<?php
namespace Rindow\NeuralNetworks\Loss;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Layer\Layer;
use Rindow\NeuralNetworks\Layer\AbstractLayer;
use InvalidArgumentException;
use DomainException;

abstract class AbstractCrossEntropy extends AbstractLayer implements Loss,Layer
{
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
        //extract($this->extractArgs([
        //],$options));
        $this->backend = $K = $backend;
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

    protected function call(NDArray $inputs, bool $training) : NDArray
    {
        $K = $this->backend;
        $this->outputs = $this->activationFunction($inputs);
        return $this->outputs;
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        if($this->fromLogits) {
            return $dOutputs;
        }
        return $this->diffActivationFunction($dOutputs, $this->outputs);
    }

    public function loss(NDArray $trues, NDArray $predicts) : float
    {
        $this->trues = $trues;
        if(!$this->fromLogits) {
            $this->outputs = $predicts;
        }
        return $this->lossFunction(
            $this->trues, $predicts, $this->fromLogits);
    }

    public function differentiateLoss() : NDArray
    {
        return $this->diffLossFunction(
            $this->trues, $this->outputs, $this->fromLogits);
    }
}
