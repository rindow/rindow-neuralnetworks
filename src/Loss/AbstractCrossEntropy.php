<?php
namespace Rindow\NeuralNetworks\Loss;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Layer\Layer;
use InvalidArgumentException;
use DomainException;

abstract class AbstractCrossEntropy implements Loss
{
    protected $backend;
    protected $outputs;
    protected $trues;
    protected $fromLogits = false;

    abstract protected function activationFunction(NDArray $predicts) : NDArray;
    abstract protected function lossFunction(NDArray $trues, NDArray $predicts, bool $fromLogits) : float;
    abstract protected function deltaLossFunction(NDArray $trues, NDArray $predicts, bool $fromLogits) : NDArray;

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

    public function loss(NDArray $trues, NDArray $predicts) : float
    {
        $this->trues = $trues;
        if($this->fromLogits) {
            $this->predicts = $this->activationFunction($predicts);
        } else {
            $this->predicts = $predicts;
        }
        $this->loss = $this->lossFunction(
            $this->trues, $this->predicts, $this->fromLogits);
        return $this->loss;
    }

    public function differentiateLoss() : NDArray
    {
        return $this->deltaLossFunction(
            $this->trues, $this->predicts, $this->fromLogits);
    }

}
