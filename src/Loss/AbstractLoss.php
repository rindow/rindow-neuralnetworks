<?php
namespace Rindow\NeuralNetworks\Loss;

use Interop\Polite\Math\Matrix\NDArray;
//use Rindow\NeuralNetworks\Activation\Activation;
use Rindow\NeuralNetworks\Gradient\Core\Variable;
use Rindow\NeuralNetworks\Gradient\Core\GradientTape;
use Rindow\NeuralNetworks\Gradient\Core\GradientUtils;
use InvalidArgumentException;
use DomainException;
use ArrayAccess;

abstract class AbstractLoss //implements Loss
{
    use GradientUtils;
    protected $generation;
    protected $inputsVariables;
    protected $outputsVariables;

    abstract protected function call(NDArray $trues, NDArray $predicts) : NDArray;
    abstract protected function differentiate(array $dOutputs, ArrayAccess $grads=null, array $oidsToCollect=null) : array;

    /*
    *  dinamic step interfaces
    */
    /**
    *  @return int
    */
    public function generation() : int
    {
        return $this->generation;
    }
    /**
    *  @return array<Variable>
    */
    public function inputs()
    {
        return $this->inputsVariables;
    }

    /**
    *  @return array<Variable>
    */
    public function outputs()
    {
        return $this->outputsVariables;
    }

    public function backward(array $dOutputs, ArrayAccess $grads=null, array $oidsToCollect=null) : array
    {
        $K = $this->backend;
        $container = $this->container();
        $dInputs = $this->differentiate($dOutputs, $grads, $oidsToCollect);
        //array_unshift($dInputs, $K->zeros($container->truesShape,$container->truesDtype));
        array_unshift($dInputs,null);
        return $dInputs;
    }

    public function __invoke(...$args)
    {
        return $this->forward(...$args);
    }

    /**
    *  @param array<Variable>  $inputs
    *       inputs
    *  @return array<Variable>
    *       outputs
    */
    public function forward(NDArray $trues, NDArray $predicts) : Variable
    {
        $K = $this->backend;
        [$trues,$rawTrues] = $this->packAndUnpackVariable($K,$trues);
        [$predicts,$rawPredicts] = $this->packAndUnpackVariable($K,$predicts);
        $session = $this->preGradientProcessOnSession([$trues,$predicts]);
        $session->begin();
        try {
            $container = $this->container();
            $container->truesShape = $rawTrues->shape();
            $container->truesDtype = $rawTrues->dtype();
            $loss = $this->call($rawTrues,$rawPredicts);
            $rawOutputs = $this->packVariable($K,$loss);
        } finally {
            $session->end();
        }
        $outputs = $this->postGradientProcessOnSession(
            $this->backend, $session, [$trues,$predicts], [$rawOutputs]);
        return $outputs[0];
    }

    /**
     * Call from SessionFunc in compiled graph
     */
    public function _rawCall(array $inputs,array $options)
    {
        $K = $this->backend;
        [$trues, $predicts] = $inputs;
        $container = $this->container();
        $container->truesShape = $trues->shape();
        $container->truesDtype = $trues->dtype();
        $loss = $this->call($trues,$predicts);
        return [$loss];
    }
}
