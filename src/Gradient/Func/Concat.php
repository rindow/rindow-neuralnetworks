<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use InvalidArgumentException;
use LogicException;
use Rindow\NeuralNetworks\Gradient\Scalar as ScalarInterface;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;
use Rindow\NeuralNetworks\Gradient\Core\Scalar;
use Interop\Polite\Math\Matrix\NDArray;

class Concat extends AbstractFunction
{
    protected int $numOfInputs = 1;

    protected ?int $axis;

    public function __construct(
        object $backend,
        int $numOfInputs,
        ?int $axis=null,
        ?string $name=null,
    )
    {
        parent::__construct($backend,name:$name);

        if($numOfInputs <= 0) {
            throw new InvalidArgumentException('inputs must not be empty.');
        }
        $this->numOfInputs = $numOfInputs;
        $axis ??= -1;
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
        $shapes = array_reduce($inputs,function($shapes,$input) {
            $shapes[] = $input->shape();
            return $shapes;
        },[]);
        $container->shapes = $shapes;
        $container->axis = $this->axis;
        $outputs = $K->concat($inputs,axis:$this->axis);
        return [$outputs];
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
        $container = $this->container();

        $axis = $container->axis;
        if($axis<0) {
            $axis = count($container->shapes[0])+$axis;
        }
        $sizeSplits = [];
        foreach($container->shapes as $shape) {
            $sizeSplits[] = $shape[$axis];
        }

        $dInputs = $K->split($dOutputs[0],$sizeSplits,axis:$axis);

        return $dInputs;
    }
}
