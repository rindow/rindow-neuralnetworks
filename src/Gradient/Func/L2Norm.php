<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;

class L2Norm extends AbstractFunction
{
    protected ?int $axis;
    
    public function __construct(
        object $backend,
        ?int $axis=null,
        ?string $name=null,
    )
    {
        parent::__construct($backend,name:$name);
        $this->axis = $axis;
    }

    protected function call(array $inputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        $container->input = $inputs[0];
        //$var = $K->sum($K->square($inputs[0]),axis:$this->axis,ndarray:true);
        //$output = $K->sqrt($var);
        $output = $K->l2norm($inputs[0],axis:$this->axis);
        $container->output = $output;
        return [$output];
    }

    protected function reshapeFlatArray(NDArray $x) : NDArray
    {
        $shape = $x->shape();
        $feature = array_pop($shape);
        return $x->reshape([array_product($shape),$feature]);
    }

    protected function differentiate(array $dOutputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        $input = $container->input;
        $output = $container->output;
        $axis = $this->axis;

        $dInput = $K->dL2norm($dOutputs[0],$input,$output,axis:$axis);
        //if($axis === null || $axis === 0) {
        //    // broardcast on axis 0
        //    $dInput = $K->mul($dOutputs[0],$K->div($input,$output));
        //} elseif($axis === -1 || $axis === $input->ndim()-1) {
        //    // broardcast on last axis
        //    $input = $this->reshapeFlatArray($input);
        //    $output = $output->reshape([$output->size()]);
        //    $dOutput = $dOutputs[0]->reshape([$dOutputs[0]->size()]);
        //    $dInput = $K->mul($K->div($input,$output,trans:true),$dOutput,trans:true);
        //    $dInput = $dInput->reshape($container->input->shape());
        //} else {
        //    throw new InvalidArgumentException("Unsupported axis: {$axis}");
        //}
        //if($axis===null) {
        //    $dInput = $dInput->reshape($input->shape());
        //}
        return [$dInput];
    }
}
