<?php
namespace Rindow\NeuralNetworks\Loss;

use Interop\Polite\Math\Matrix\NDArray;
//use Rindow\NeuralNetworks\Activation\Activation;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Gradient\Core\GradientTape;
use Rindow\NeuralNetworks\Gradient\Core\GradientUtils;
use InvalidArgumentException;
use DomainException;
use ArrayAccess;

abstract class AbstractLoss implements Loss
{
    use GradientUtils;

    //protected int $generation;
    ///** @var array<Variable> */
    //protected array $inputsVariables;
    ///** @return array<Variable> */
    //protected array $outputsVariables;

    abstract protected function call(NDArray $trues, NDArray $predicts) : NDArray;

    /**
     * @param array<NDArray> $dOutputs
     * @param ArrayAccess<object,object> $grads
     * @param array<NDArray> $oidsToCollect
     * @return array<NDArray>
     */
    abstract protected function differentiate(array $dOutputs, ArrayAccess $grads=null, array $oidsToCollect=null) : array;

    protected object $backend;
    protected bool $fromLogits = false;
    protected string $reduction = 'sum';

    public function __construct(
        object $backend,
        bool $from_logits=null,
        string $reduction=null,
        )
    {
        // defaults
        $from_logits = $from_logits ?? false;
        $reduction = $reduction ?? 'sum';

        $this->backend = $backend;
        $this->fromLogits = $from_logits;
        $this->reduction = $reduction;
    }

    public function setFromLogits(bool $fromLogits) : void
    {
        $this->fromLogits = $fromLogits;
    }

    public function fromLogits() : bool
    {
        return $this->fromLogits;
    }

    public function getConfig() : array
    {
        return [
        ];
    }

    /*
    * temporary disabled but rerise when it is used the RT
    *  dinamic step interfaces
    */
    ///**
    //*  @return int
    //*/
    //public function generation() : int
    //{
    //    return $this->generation;
    //}

    ///**
    //*  @return array<Variable>
    //*/
    //public function inputs() : array
    //{
    //    return $this->inputsVariables;
    //}

    ///**
    //*  @return array<Variable>
    //*/
    //public function outputs() : array
    //{
    //    return $this->outputsVariables;
    //}

    /**
     * @return array{NDArray,NDArray}
     */
    protected function flattenShapes(NDArray $trues, NDArray $predicts) : array
    {
        $origTrueShape = $trues->shape();
        $origPredictsShape = $predicts->shape();
        if($trues->ndim()<$predicts->ndim()) {
            $shape = $trues->shape();
            array_push($shape,1);
            $trues = $trues->reshape($shape);
        }
        if($trues->shape()!=$predicts->shape()){
            throw new InvalidArgumentException('trues and predicts must be same shape of dimensions. '.
                'trues,predicts are ['.implode(',',$origTrueShape).'],['.implode(',',$predicts->shape()).']');
        }
        if($predicts->ndim()==1) {
            $size = $predicts->size();
            $predicts = $predicts->reshape([1,$size]);
            $trues = $trues->reshape([1,$size]);
        }
        //$origPredictsShape = $predicts->shape();
        //$orgTruesShape = $trues->shape();
        $batchShape = $predicts->shape();
        $feature = array_pop($batchShape);
        $batchSize = array_product($batchShape);
        $container = $this->container();
        $container->predictsShape = $origPredictsShape;
        $container->batchShape = $batchShape;
        $container->batchSize = $batchSize;
        $container->batchShape = $batchShape;
        $container->feature = $feature;
        $trues = $trues->reshape([$batchSize,$feature]);
        $predicts = $predicts->reshape([$batchSize,$feature]);
        return [$trues,$predicts];
    }

    protected function reshapeLoss(NDArray $loss) : NDArray
    {
        $container = $this->container();
        if($this->reduction=='none') {
            $loss = $loss->reshape($container->batchShape);
        }
        return $loss;
    }

    protected function flattenLoss(float|NDArray $loss) : NDArray
    {
        $container = $this->container();
        if($this->reduction=='none') {
            if(($loss instanceof NDArray)&&$loss->ndim()>0) {
                $loss = $loss->reshape([$container->batchSize]);
            } else {
                throw new InvalidArgumentException('loss must not be scaler. ');
            }
        }
        return $loss;
    }

    protected function reshapePredicts(NDArray $predicts) : NDArray
    {
        $container = $this->container();
        $predicts = $predicts->reshape($container->predictsShape);
        return $predicts;
    }

    /**
     * @return array{NDArray,NDArray}
     */
    protected function flattenShapesForSparse(NDArray $trues, NDArray $predicts) : array
    {
        $origTrueShape = $trues->shape();
        $origPredictsShape = $trues->shape();

        $batchShape = $predicts->shape();
        $feature = array_pop($batchShape);
        $batchSize = array_product($batchShape);
        if($trues->shape()!=$batchShape){
            throw new InvalidArgumentException('trues and predicts must be same batch-shape of dimensions. '.
                'trues,predicts are ['.implode(',',$origTrueShape).'],['.implode(',',$origPredictsShape).']');
        }

        $container = $this->container();
        $container->predictsShape = $predicts->shape();
        $container->batchShape = $batchShape;
        $container->batchSize = $batchSize;
        $container->batchShape = $batchShape;
        $container->feature = $feature;
        $trues = $trues->reshape([$batchSize]);
        $predicts = $predicts->reshape([$batchSize,$feature]);
        return [$trues,$predicts];
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

    public function __invoke(mixed ...$args) : mixed
    {
        return $this->forward(...$args);
    }

    /**
    *  @param Variable  $trues
    *  @param Variable  $predicts
    */
    public function forward(NDArray $trues, NDArray $predicts) : Variable
    {
        $K = $this->backend;
        [$trues,$rawTrues] = $this->packAndUnpackVariable($K,$trues);
        [$predicts,$rawPredicts] = $this->packAndUnpackVariable($K,$predicts);
        $session = $this->preGradientProcessOnSession([$trues,$predicts]);
        $session->begin();
        try {
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
     * @param array<NDArray> $inputs
     * @param array<string,mixed> $options
     * @return array<NDArray>
     */
    public function _rawCall(array $inputs, array $options) : array
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
