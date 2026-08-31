<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;

class ReduceMax extends AbstractFunction
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
        $max = $K->max($inputs[0],axis:$this->axis,keepdims:$this->keepdims);
        if(!($max instanceof NDArray)) {
            $max = $K->array($max);
        }
        return [$max];
    }

    protected function differentiate(array $dOutputs) : array
    {
        $K = $this->backend;
        //echo "===max===\n";
        //echo 'dOutputs='.$K->toString($dOutputs[0])."\n";
        $container = $this->container();
        $x = $container->inputs[0];
        $axis = $this->axis;
        $shape = $x->shape();
        if($axis!==null) {
            if($axis<0) {
                $axis += $x->ndim();
            }
        }
        $argMax = $K->argMax($x,axis:$axis,dtype:NDArray::int32);
        if(is_numeric($argMax)) {
            $argMax = $K->array($argMax,dtype:NDArray::int32);
        }

        //echo "argMax=".$K->toString($argMax)."\n";
        //echo "dOutputs=".$K->toString($dOutputs[0])."\n";
        //echo "shape=".$K->shapeToString($shape)."\n";
        $dInputs = $K->scatterb(
            $argMax,                    // indices
            $dOutputs[0],               // updates
            $shape,                     // shape
            axis:$axis,
            batchDims:$axis,
            detailDepth:$x->ndim(),
            indexDepth:$axis,
        );

        //echo 'dInputs='.$K->toString($dInputs)."\n";
        //echo 'dInputs=['.implode(',',array_map(fn($x)=>'['.implode(',',$x).']',$dInput->toArray()))."]\n";
        return [$dInputs];
    }
}
