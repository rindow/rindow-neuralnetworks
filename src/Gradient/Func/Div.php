<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;
use Rindow\NeuralNetworks\Gradient\Core\NullValue;
use Rindow\NeuralNetworks\Gradient\Core\Scalar;
use Rindow\NeuralNetworks\Gradient\Scalar as ScalarInterface;
use InvalidArgumentException;

class Div extends AbstractFunction
{
    protected int $numOfInputs = 2;

    protected function preprocess(array $inputs) : array
    {
        if(is_numeric($inputs[0])) {
            $inputs[0] = new Scalar($inputs[0]);
        }
        if(is_numeric($inputs[1])) {
            $inputs[1] = new Scalar($inputs[1]);
        }
        if($inputs[0] instanceof ScalarInterface && $inputs[1] instanceof ScalarInterface) {
            throw new InvalidArgumentException("A scalar cannot be specified for both inputs.");
        }
        return $inputs;
    }

    protected function call(array $inputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        $container->inputs = $inputs;
        if($inputs[0] instanceof ScalarInterface) {
            if($inputs[0]->value()==0) {
                $output = $K->zerosLike($inputs[0]);
            } else {
                $output = $K->reciprocal($inputs[1], alpha:(1/$inputs[0]->value()));
            }
        } elseif($inputs[1] instanceof ScalarInterface) {
            if($inputs[1]->value()==0) {
                $output = $K->fill($inputs[0]->shape(),NAN);
            } else {
                $output = $K->scale(1/$inputs[1]->value(), $inputs[0]);
            }
        } else {
            $output = $K->div($inputs[0],$inputs[1]);
        }
        return [$output];
    }

    protected function differentiate(array $dOutputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        [$x0, $x1] = $container->inputs;

        if($x0 instanceof ScalarInterface) {
            $dx0 = new NullValue();
            if($x0->value()==0) {
                $dx1 = $K->zeroLike($x1);
            } else {
                $dx1 = $K->mul($dOutputs[0],$K->reciprocal($K->square($x1),alpha:1/(-1*$x0->value())));
            }
        } else if($x1 instanceof ScalarInterface) {
            $dx0 = $K->scale($x1->value(), $dOutputs[0]);
            $dx1 = new NullValue();
        } else {
            $dx0 = $K->div($dOutputs[0], $x1);
            $dx1 = $K->mul($dOutputs[0],$K->div($K->scale(-1,$x0),$K->square($x1)));
            // for broadcasted inputs
            if($x0->ndim() != $dx0->ndim()) {
                $dx0 = $K->sum($dx0, axis:0);
            }
            if($x1->ndim() != $dx1->ndim()) {
                $dx1 = $K->sum($dx1, axis:0);
            }
        }
        return [$dx0, $dx1];
    }
}
