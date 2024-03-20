<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;

class Div extends AbstractFunction
{
    protected $numOfInputs = 2;

    protected function call(array $inputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        $container->inputs = $inputs;
        $output = $K->div($inputs[0],$inputs[1]);
        return [$output];
    }

    protected function differentiate(array $dOutputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        [$x0, $x1] = $container->inputs;

        $dx0 = $K->div($dOutputs[0], $x1);
        $dx1 = $K->mul($dOutputs[0],$K->div($K->scale(-1,$x0),$K->square($x1)));

        // for broadcasted inputs
        if($x0->ndim() != $dx0->ndim()) {
            $dx0 = $K->sum($dx0, axis:0);
        }
        if($x1->ndim() != $dx1->ndim()) {
            $dx1 = $K->sum($dx1, axis:0);
        }
        return [$dx0, $dx1];
    }
}
