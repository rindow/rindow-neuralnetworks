<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;
use InvalidArgumentException;

class Minimum extends AbstractFunction
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
            throw new InvalidArgumentException('Number of dimension variable #1 must be greater than variable #2 or equals.');
        }
        $output = $this->backend->minimum($a,$x);
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

        $da = $K->mul($dOutputs[0],$K->lessEqual($a,$x));
        $dx = $K->mul($dOutputs[0],$K->greater($a,$x));

        // for broadcasted inputs
        if($x->ndim() != $dx->ndim()) {
            $dx = $K->sum($dx, axis:0);
        }
        return [$da, $dx];
    }
}
