<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;

class Mul extends AbstractFunction
{
    protected $numOfInputs = 2;

    protected function call(array $inputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        $container->inputs = $inputs;
        $output = $K->mul($inputs[0],$inputs[1]);
        return [$output];
    }

    protected function differentiate(array $dOutputs) : array
    {
        //echo "===mul===\n";
        //echo 'dOutputs='.implode(',',$dOutputs[0]->toArray())."\n";
        $K = $this->backend;
        //$inputs = $this->inputsVariables;
        $container = $this->container();
        [$x0, $x1] = $container->inputs;

        $dx0 = $K->mul($dOutputs[0], $x1);
        $dx1 = $K->mul($dOutputs[0], $x0);

        // for broadcasted inputs
        if($x0->ndim() != $dx0->ndim()) {
            $dx0 = $K->sum($dx0, axis:0);
        }
        if($x1->ndim() != $dx1->ndim()) {
            $dx1 = $K->sum($dx1, axis:0);
        }
        //echo 'dx0=['.implode(',',$dx0->toArray())."]\n";
        //echo 'dx1=['.implode(',',$dx1->toArray())."]\n";
        return [$dx0, $dx1];
    }
}
