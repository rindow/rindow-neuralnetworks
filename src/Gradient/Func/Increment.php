<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use InvalidArgumentException;
use Rindow\NeuralNetworks\Gradient\Scalar as ScalarInterface;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;
use Rindow\NeuralNetworks\Gradient\Core\Scalar;
use Interop\Polite\Math\Matrix\NDArray;

class Increment extends AbstractFunction
{
    protected $numOfInputs = 3;

    protected function preprocess(array $inputs) : array
    {
        if(is_numeric($inputs[1])) {
            $inputs[1] = new Scalar($inputs[1]);
        }
        if(is_numeric($inputs[2])) {
            $inputs[2] = new Scalar($inputs[2]);
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

        $array = $inputs[0];
        $beta  = $inputs[1];
        $alpha = $inputs[2];
        $beta = $this->toScalar($beta,2);
        $alpha = $this->toScalar($alpha,3);
        $container->alpha = $alpha;
        $container->beta = $beta;

        //  output = a*X + b
        $output = $K->increment($array,$beta,$alpha);
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
        [$array, $beta, $alpha] = $container->inputs;

        // db = sum(dOutput) 
        if($beta instanceof ScalarInterface) {
            $dBeta = new Scalar(0); // no backward if constant
        } elseif($beta instanceof NDArray) {
            if($beta->ndim()!=0) {
                throw new InvalidArgumentException('arg #1 must not be scalar.');
            }
            $dBeta = $K->sum($dOutputs[0]);
            if(!($dBeta instanceof NDArray)) {
                $dBeta = $K->array($dBeta);
            }
        }

        // da = sum(dOut * X)
        if($alpha instanceof ScalarInterface) {
            $dAlpha = new Scalar(0);
        } elseif($alpha instanceof NDArray) {
            if($alpha->ndim()!=0) {
                throw new InvalidArgumentException('arg #1 must not be scalar.');
            }
            $dAlpha = $K->sum($K->mul($dOutputs[0],$array));
            if(!($dAlpha instanceof NDArray)) {
                $dAlpha = $K->array($dAlpha);
            }
        }
        $alpha = $container->alpha;

        $dInputs = $K->scale($alpha,$dOutputs[0]);
        
        return [$dInputs, $dBeta, $dAlpha];
    }
}
