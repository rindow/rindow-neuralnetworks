<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;
use Rindow\NeuralNetworks\Gradient\Core\NullValue;
use Rindow\NeuralNetworks\Gradient\Core\Scalar;
use Rindow\NeuralNetworks\Gradient\Scalar as ScalarInterface;
use InvalidArgumentException;

class Sub extends AbstractFunction
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
            $output = $K->increment($inputs[1], beta:$inputs[0]->value(), alpha:-1);
        } elseif($inputs[1] instanceof ScalarInterface) {
            $output = $K->increment($inputs[0], beta:-1*$inputs[1]->value());
        } else {
            $output = $K->sub($inputs[0],$inputs[1]);
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
            $dx1 = $K->scale(-1,$dOutputs[0]);
        } else if($x1 instanceof ScalarInterface) {
            $dx0 = $dOutputs[0];
            $dx1 = new NullValue();
        } else {
            $dx0 = $dOutputs[0];
            $dx1 = $K->scale(-1,$dOutputs[0]);
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
