<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;

class ReduceMean extends AbstractFunction
{
    public function __construct(
        object $backend,
        int $axis=null,
    )
    {
        parent::__construct($backend);
        $this->axis = $axis;
    }

    protected function call(array $inputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        $container->inputs = $inputs;
        $mean = $K->mean($inputs[0],$this->axis);
        if(!($mean instanceof NDArray)) {
            $mean = $this->backend->array($mean);
        }
        return [$mean];
    }

    protected function differentiate(array $dOutputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        $x = $container->inputs[0];
        $axis = $this->axis;
        if($axis===null) {
            $n = $x->size();
        } else {
            if($axis<0) {
                $axis = $ndim+$x->ndim();
            }
            $shape = $x->shape();
            $n = $shape[$axis];
        }
        $dSum = $K->scale(1/$n,$dOutputs[0]);
        $dInput = $K->repeat($dSum,$n,$axis);
        if($axis===null) {
            $dInput = $dInput->reshape($x->shape());
        }
        return [$dInput];
    }
}
