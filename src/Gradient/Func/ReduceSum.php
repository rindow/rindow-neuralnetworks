<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;

class ReduceSum extends AbstractFunction
{
    protected ?int $axis;
    protected ?bool $keepdims;
    
    public function __construct(
        object $backend,
        ?int $axis=null,
        ?bool $keepdims=null,
        ?string $name=null,
    )
    {
        parent::__construct($backend,name:$name);
        $this->axis = $axis;
        $this->keepdims = $keepdims;
    }

    protected function call(array $inputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        $container->inputs = $inputs;
        $sum = $K->sum($inputs[0],axis:$this->axis,keepdims:$this->keepdims);
        if(!($sum instanceof NDArray)) {
            $sum = $K->array($sum);
        }
        return [$sum];
    }

    protected function differentiate(array $dOutputs) : array
    {
        //echo "===sum===\n";
        //echo 'dOutputs='.implode(',',$dOutputs[0]->toArray())."\n";
        $K = $this->backend;
        $container = $this->container();
        $x = $container->inputs[0];
        $axis = $this->axis;
        if($axis===null) {
            $n = $x->size();
        } else {
            if($axis<0) {
                $axis += $x->ndim();
            }
            $shape = $x->shape();
            $n = $shape[$axis];
        }
        $dInput = $K->repeat($dOutputs[0],$n,axis:$axis,keepdims:$this->keepdims);
        if($axis===null) {
            $dInput = $dInput->reshape($x->shape());
        }
        //echo 'dInputs='.implode(',',$dInput->toArray())."\n";
        //echo 'dInputs=['.implode(',',array_map(fn($x)=>'['.implode(',',$x).']',$dInput->toArray()))."]\n";
        return [$dInput];
    }
}
