<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;

class ClipByValue extends AbstractFunction
{
    protected float $min;
    protected float $max;

    public function __construct(
        object $backend,
        float $min,
        float $max,
    )
    {
        parent::__construct($backend);
        $this->min = $min;
        $this->max = $max;
    }

    protected function call(array $inputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        $container->inputs = $inputs;
        $output = $K->maximum($inputs[0],$this->min);
        $output = $K->minimum($output,$this->max);
        return [$output];
    }

    protected function differentiate(array $dOutputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        $x = $container->inputs[0];
        $mask = $K->mul($K->greater($x,$this->min),$K->less($x,$this->max));
        $dInput = $K->mul($dOutputs[0],$mask);
        return [$dInput];
    }
}
