<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;

class Mul extends AbstractFunction
{
    protected $numOfInputs = 2;

    protected function call(array $inputs) : array
    {
        $K = $this->backend;
        $output = $K->mul($inputs[0],$inputs[1]);
        return [$output];
    }

    protected function differentiate(array $dOutputs) : array
    {
        $K = $this->backend;
        $inputs = $this->inputsVariables;
        [$x0, $x1] = $inputs;
        $dx0 = $K->mul($dOutputs[0], $x1->value());
        $dx1 = $K->mul($dOutputs[0], $x0->value());
        // for broadcasted inputs
        if($inputs[0]->value()->ndim() != $dx0->ndim()) {
            $dx0 = $K->sum($dx0, $axis=0);
        }
        if($inputs[1]->value()->ndim() != $dx1->ndim()) {
            $dx1 = $K->sum($dx1, $axis=0);
        }
        return [$dx0, $dx1];
    }
}
