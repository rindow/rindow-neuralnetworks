<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;
use Rindow\NeuralNetworks\Support\GenericUtils;

class Matmul extends AbstractFunction
{
    use GenericUtils;
    protected $numOfInputs = 2;
    protected $transA;
    protected $transB;

    public function __construct($backend, array $options=null)
    {
        extract($this->extractArgs([
            'transpose_a'=>false,
            'transpose_b'=>false,
        ],$options));
        parent::__construct($backend, $options);
        $this->transA = $transpose_a;
        $this->transB = $transpose_b;
    }

    protected function call(array $inputs) : array
    {
        $K = $this->backend;
        [$x0, $x1] = $inputs;
        $outputs = $K->matmul($x0,$x1,$this->transA,$this->transB);
        return [$outputs];
    }

    protected function differentiate(array $dOutputs) : array
    {
        $K = $this->backend;
        $inputs = $this->inputsVariables;
        [$x0, $x1] = $inputs;

        $dx0 = $K->matmul($dOutputs[0], $x1->value(),$this->transB,!$this->transB);
        if($this->transA) {
            if($dx0->ndim()<3) {
                $dx0 = $K->transpose($dx0);
            } else {
                $dx0 = $K->batch_transpose($dx0);
            }
        }
        $dx1 = $K->matmul($x0->value(), $dOutputs[0],!$this->transA,$this->transA);
        if($this->transB) {
            if($dx1->ndim()<3) {
                $dx1 = $K->transpose($dx1);
            } else {
                $dx1 = $K->batch_transpose($dx1);
            }
        }
        return [$dx0, $dx1];
    }
}
