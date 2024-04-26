<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use InvalidArgumentException;
use LogicException;
use Rindow\NeuralNetworks\Gradient\Scalar as ScalarInterface;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;
use Rindow\NeuralNetworks\Gradient\Core\Scalar;
use Interop\Polite\Math\Matrix\NDArray;

class Scale extends AbstractFunction
{
    protected int $numOfInputs = 2;

    protected function preprocess(array $inputs) : array
    {
        if(is_numeric($inputs[0])) {
            $inputs[0] = new Scalar($inputs[0]);
        }
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
        $container = $this->container();
        $container->inputs = $inputs;

        $alpha = $inputs[0];
        $array = $inputs[1];
        $alpha = $this->toScalar($alpha,1);
        $container->alpha = $alpha;

        $output = $K->scale($alpha,$array);
        return [$output];
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
        [$alpha, $array] = $container->inputs;

        if($alpha instanceof ScalarInterface) {
            $dAlpha = new Scalar(0);
        } elseif($alpha instanceof NDArray) {
            if($alpha->ndim()!=0) {
                throw new InvalidArgumentException('arg #1 must not be scalar.');
            }
            // da = sum(dOut * X)
            $dAlpha = $K->sum($K->mul($dOutputs[0],$array));
            if(!($dAlpha instanceof NDArray)) {
                $dAlpha = $K->array($dAlpha);
            }
        } else {
            throw new LogicException('alpha must Scalar or NDArray');
        }
        // dX = a * dOut
        $alpha = $container->alpha;
        $dInputs = $K->scale($alpha,$dOutputs[0]);
        
        return [$dAlpha, $dInputs];
    }
}
