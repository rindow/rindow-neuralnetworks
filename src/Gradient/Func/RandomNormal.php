<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;
use Rindow\NeuralNetworks\Gradient\Core\NullValue;

class RandomNormal extends AbstractFunction
{
    protected float $mean;
    protected float $scale;
    /** @var array<int> $batchShape */
    protected array $batchShape;
    protected ?int $seed;

    /**
     * @param array<int>|null $batchShape
     */
    public function __construct(
        object $backend,
        ?float $mean=null,
        ?float $scale=null,
        ?array $batchShape=null,
        ?int $seed=null,
        ?string $name=null,
    )
    {
        $mean ??= 0.0;
        $scale ??= 1.0;
        $batchShape ??= [];
        parent::__construct($backend,name:$name);
        $this->mean = $mean;
        $this->scale = $scale;
        $this->batchShape = $batchShape;
        $this->seed = $seed;
    }

    protected function call(array $inputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        $shape = $inputs[0]->shape();
        $dtype = $inputs[0]->dtype();
        $shape = array_merge($this->batchShape,$shape);
        $outputs = $K->randomNormalVariables($shape,$this->mean,$this->scale,dtype:$dtype,seed:$this->seed);
        $container->shape = $shape;
        $container->dtype = $dtype;
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
