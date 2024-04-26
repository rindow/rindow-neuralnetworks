<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;
use Rindow\NeuralNetworks\Gradient\Core\Scalar;

class Repeat extends AbstractFunction
{
    protected int $numOfInputs = 2;

    protected ?int $axis;
    protected ?bool $keepdims;
    
    public function __construct(
        object $backend,
        int $axis=null,
        bool $keepdims=null,
    )
    {
        parent::__construct($backend);
        $this->axis = $axis;
        $this->keepdims = $keepdims;
    }

    protected function preprocess(array $inputs) : array
    {
        if(is_numeric($inputs[1])) {
            $inputs[1] = new Scalar($inputs[1]);
        }
        return $inputs;
    }

    protected function call(array $inputs) : array
    {
        $K = $this->backend;
        $x = $inputs[0];
        $repeats = $inputs[1];
        $repeats = $this->toScalar($repeats,1);

        $container = $this->container();
        $container->inpShape = $x->shape();
        $output = $K->repeat($x,$repeats,axis:$this->axis,keepdims:$this->keepdims);
        if($this->keepdims) {
            $container->repeats = $repeats;
            $container->outShape = $output->shape();
        }
        return [$output];
    }

    protected function differentiate(array $dOutputs) : array
    {
        $K = $this->backend;
        $dy = $dOutputs[0];

        $container = $this->container();
        $shape = $container->inpShape;
        $axis = $this->axis;
        if($axis===null) {
            $axis = 0;
        } elseif($axis<0) {
            $axis += count($shape);
        }
        if($this->keepdims||$this->axis===null) {
            $repeats = $container->repeats;
            $outerShape = array_slice($shape,0,$axis);
            $innerShape = array_slice($shape,$axis);
            $shape = array_merge($outerShape,[$repeats],$innerShape);
            $dy = $dy->reshape($shape);
        }
        $dInput = $K->sum($dy,axis:$axis);

        $dRepeats = new Scalar(0);
        return [$dInput,$dRepeats];
    }
}
