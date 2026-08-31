<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;
use InvalidArgumentException;

class Maximum extends AbstractFunction
{
    protected int $numOfInputs = 2;

    /**
    *  @param array<NDArray>  $inputs
    *  @return array<NDArray>
    */
    protected function call(array $inputs) : array
    {
        $container = $this->container();
        $container->inputs = $inputs;
        [$a,$x] = $inputs;

        if($a->ndim() < $x->ndim()) {
            $aNdim = $a->ndim();
            $xNdim = $x->ndim();
            throw new InvalidArgumentException("Number of dimension variable #1 must be greater than variable #2 or equals. ndim #1 is $aNdim, ndim #2 is $xNdim)");
        }
        $output = $this->backend->maximum($inputs[0],$inputs[1]);
        return [$output];
    }

    /**
    *  @param array<NDArray>  $dOutputs
    *  @return array<NDArray>
    */
    protected function differentiate(array $dOutputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        [$a, $x] = $container->inputs;

        $da = $K->mul($dOutputs[0],$K->greaterEqual($a,$x));
        $dx = $K->mul($dOutputs[0],$K->less($a,$x));

        // for broadcasted inputs
        if($x->ndim() != $dx->ndim()) {
            $dx = $K->sum($dx, axis:0);
        }
        return [$da, $dx];
    }
}
