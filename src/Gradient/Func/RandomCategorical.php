<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;
use Rindow\NeuralNetworks\Gradient\Core\NullValue;

class RandomCategorical extends AbstractFunction
{
    /** @var array<int> $batchShape */
    protected array $batchShape;
    protected bool $softmax;
    protected int $dtype;
    protected ?int $seed;

    /**
     * @param array<int>|null $batchShape
     */
    public function __construct(
        object $backend,
        ?array $batchShape=null,
        ?bool $softmax=null,
        ?int $dtype=null,
        ?int $seed=null,
        ?string $name=null,
    )
    {
        $softmax ??= true;
        $dtype ??= NDArray::int32;
        $batchShape ??= [];
        parent::__construct($backend,name:$name);
        $this->batchShape = $batchShape;
        $this->softmax = $softmax;
        $this->dtype = $dtype;
        $this->seed = $seed;
    }

    protected function call(array $inputs) : array
    {
        $K = $this->backend;
        $container = $this->container();

        $logits = $inputs[0];
        $batchShape = $logits->shape();
        if(count($batchShape)==0) {
            throw new InvalidArgumentException("More than one dimension is required.");
        }
        $singleLogits = (count($batchShape)==1);
        $numActions = array_pop($batchShape);
        $batchDims = count($batchShape);
        $batchSize = array_product($batchShape);
        $logits = $logits->reshape([$batchSize,$numActions]);
        if($this->softmax) {
            $probs = $K->softmax($logits);
        } else {
            $probs = $K->exp($logits);
        }
        $numSamples = null;
        if(count($this->batchShape)>0) {
            if(!$singleLogits) {
                throw new InvalidArgumentException("If batchShape is specified, the input must be one-dimensional.");
            }
            $probs = $probs->reshape([$probs->size()]);
            $numSamples = array_product($this->batchShape);
            $batchShape = $this->batchShape;
        }
        $outputs = $K->randomCategorical(
            $probs,
            numSamples:$numSamples,
            dtype:$this->dtype,
            seed:$this->seed
        );
        $outputs = $outputs->reshape($batchShape);
        $this->unbackpropagatables = [true];
        return [$outputs];
    }

    protected function differentiate(array $dOutputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        $dInputs = [new NullValue()];
        return $dInputs;
    }
}
