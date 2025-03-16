<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;

class Matmul extends AbstractFunction
{
    protected int $numOfInputs = 2;
    protected bool $transA;
    protected bool $transB;

    public function __construct(
        object $backend,
        ?bool $transpose_a=null,
        ?bool $transpose_b=null,
    )
    {
        $transpose_a = $transpose_a ?? false;
        $transpose_b = $transpose_b ?? false;

        parent::__construct($backend);
        $this->transA = $transpose_a;
        $this->transB = $transpose_b;
    }

    protected function call(array $inputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        $container->inputs = $inputs;
        [$x0, $x1] = $inputs;
        $outputs = $K->matmul($x0,$x1,$this->transA,$this->transB);
        return [$outputs];
    }

    protected function differentiate(array $dOutputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        [$x0, $x1] = $container->inputs;

        $dx0 = $K->matmul($dOutputs[0], $x1,$this->transB,!$this->transB);
        if($this->transA) {
            if($dx0->ndim()<3) {
                $dx0 = $K->transpose($dx0);
            } else {
                $dx0 = $K->batch_transpose($dx0);
            }
        }
        $dx1 = $K->matmul($x0, $dOutputs[0],!$this->transA,$this->transA);
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
