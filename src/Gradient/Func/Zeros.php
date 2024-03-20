<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use InvalidArgumentException;
use Rindow\NeuralNetworks\Gradient\ArrayShape as ArrayShapeInterface;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Gradient\Scalar as ScalarInterface;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;
use Rindow\NeuralNetworks\Gradient\Core\Scalar;
use Interop\Polite\Math\Matrix\NDArray;

class Zeros extends AbstractFunction
{
    protected $numOfInputs = 1;
    protected ?int $dtype;

    public function __construct(
        object $backend,
        int $dtype=null,
    )
    {
        parent::__construct($backend);
        $this->dtype = $dtype;
    }

    protected function preprocess(array $inputs) : array
    {
        $inputs = $this->extractShapeArgment($inputs[0]);
        return $inputs;
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
        $shape = $this->translateToShape($inputs);
        $this->unbackpropagatables = [true];
        return [$K->zeros($shape,dtype:$this->dtype)];
    }

    /**
    *  @param array<NDArray>  $dOutputs
    *       difference outputs
    *  @return array<NDArray>
    *       difference inputs
    */
    protected function differentiate(array $dOutputs) : array
    {
        throw new LogicException('This function is not differentiable');
    }
}
