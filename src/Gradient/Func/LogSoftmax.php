<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;

class LogSoftmax extends AbstractFunction
{
    protected function call(array $inputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        $container->inputs = $inputs;
        $x = $inputs[0];

        if($x->ndim()<1) {
            throw new InvalidArgumentException('The input of LogSoftmax must be a vector.');
        }
        $dtype = $x->dtype();
        $shape = $x->shape();
        $flattenShape = $shape;
        if(count($shape)>1) {
            $batchSize = array_product(array_splice($flattenShape,0,-1));
            array_unshift($flattenShape,$batchSize);
        }
        $x = $x->reshape($flattenShape);
        $max_x = $K->max($x, axis:-1);
        $x_shifted = $K->sub($x, $max_x, trans:true);
        $exp_shifted = $K->exp($x_shifted);
        $sum_exp_shifted = $K->sum($exp_shifted, axis:-1);
        $log_sum_exp = $K->log($sum_exp_shifted);
        $logsumexp_term = $K->add($max_x, $log_sum_exp);
        $output = $K->sub($x, $logsumexp_term, trans:true);
    
        $container->output = $output;

        $output = $output->reshape($shape);
    
        return [$output];
    }

    protected function differentiate(array $dOutputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        $output = $container->output;
        $softmax_out = $K->exp($output);

        $shape = $dOutputs[0]->shape();
        $dout = $dOutputs[0]->reshape($softmax_out->shape());
    
        $sum_dout = $K->sum($dout, axis:-1);
        $dInput = $K->sub($dout, $K->mul($softmax_out, $sum_dout, trans:true));
        $dInput = $dInput->reshape($shape);

        return [$dInput];
    }
}
