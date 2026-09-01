<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;
use Rindow\NeuralNetworks\Gradient\Core\NullValue;
use InvalidArgumentException;

class Where extends AbstractFunction
{
    protected int $numOfInputs = 3;
    protected bool $normalize = false;

    public function __construct(
        object $backend,
        ?bool $normalize=null,
        ?string $name=null,
    )
    {
        parent::__construct($backend,name:$name);
        $this->normalize = $normalize ?? false;
    }

    /**
     * param array<NDArray> $inputs
     * return array<NDArray>
     */
    protected function call(array $inputs): array
    {
        $container = $this->container();
        $container->inputs = $inputs;

        $condition = $inputs[0]; // NDArray bool
        $x = $inputs[1];        // Select when condition is true.
        $y = $inputs[2];        // Select when condition is false.
        $K = $this->backend;

        if($x->shape()!=$y->shape()) {
            throw new InvalidArgumentException('unmatch shape of x and y');
        }
        if($x->shape()!=$condition->shape()) {
            throw new InvalidArgumentException('unmatch shape of x and condition');
        }

        $output = $K->where(
            $condition,$x,$y,
            normalize:$this->normalize
        );

        $container->condition = $condition;
        return [$output];
    }

    /**
     * param array<NDArray> $dOutputs
     * return array<NDArray>
     */
    protected function differentiate(array $dOutputs): array
    {
        $K = $this->backend;
        $container = $this->container();
        $condition = $container->condition;
        $dOutput = $dOutputs[0];

        if($condition->dtype()!=NDArray::bool) {
            $condition = $K->cast($condition,NDArray::bool);
        }
        $dX = $K->masking($condition, $dOutput);
        $dY = $K->masking($K->not($condition), $dOutput);

        $x = $container->inputs[1];
        $y = $container->inputs[2];
        $dCondition = new NullValue();

        return [$dCondition, $dX, $dY];
    }
}