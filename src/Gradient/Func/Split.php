<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use InvalidArgumentException;
use LogicException;
use Rindow\NeuralNetworks\Gradient\Scalar as ScalarInterface;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;
use Rindow\NeuralNetworks\Gradient\Core\Scalar;
use Interop\Polite\Math\Matrix\NDArray;

class Split extends AbstractFunction
{
    protected int $numOfInputs = 1;

    /** @var array<int> $sizeSplits */
    protected array $sizeSplits;
    protected ?int $axis;

    /**
     * @param array<int> $sizeSplits
     */
    public function __construct(
        object $backend,
        array $sizeSplits,
        ?int $axis=null,
    )
    {
        parent::__construct($backend);
        foreach($sizeSplits as $size) {
            if(!is_int($size)) {
                throw new InvalidArgumentException('sizeSplits must be array of integer.');
            }
        }
        $this->sizeSplits = $sizeSplits;
        $this->axis = $axis;
    }

    /**
    *  @param array<NDArray>  $inputs
    *       inputs
    *  @return array<NDArray>
    *       outputs
    */
    protected function call(array $inputs) : array
    {
        $K = $this->backend;
        $container = $this->container();

        $outputs = $K->split($inputs[0],$this->sizeSplits,axis:$this->axis);
        return $outputs;
    }

    /**
    *  @param array<NDArray>  $dOutputs
    *       difference outputs
    *  @return array<NDArray>
    *       difference inputs
    */
    protected function differentiate(array $dOutputs) : array
    {
        $K = $this->backend;

        $dInputs = $K->concat($dOutputs,axis:$this->axis);

        return [$dInputs];
    }
}
