<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;

class Add extends AbstractFunction
{
    protected $numOfInputs = 2;

    /**
    *  @param array<NDArray>  $inputs
    *       inputs
    *  @return array<NDArray>
    *       outputs
    */
    protected function call(array $inputs) : array
    {
        $output = $this->backend->add($inputs[0],$inputs[1]);
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
        $dx0 = $dOutputs[0];
        $dx1 = $dOutputs[0];
        $inputs = $this->inputsVariables;
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
