<?php
namespace RindowTest\NeuralNetworks\Gradient\Layer\DenseTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

class Test extends TestCase
{
    public function newMatrixOperator()
    {
        return new MatrixOperator();
    }

    public function newNeuralNetworks($mo)
    {
        return new NeuralNetworks($mo);
    }

    public function newBackend($nn)
    {
        return $nn->backend();
    }

    public function testMatrixValue()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x = $g->Variable($K->array([[3.0], [4.0]]));
        $layer = $nn->layers->Dense($units=5,['input_shape'=>[1]]);

        $outputs = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$x) {
                $outputs = $layer($x,true);
                return $outputs;
            }
        );
        $gradients = $tape->gradient($outputs, $layer->weights());
        //$optimizer->update($model->params(),$gradients);
        $this->assertCount(2,$gradients);
        $this->assertEquals([1,5],$gradients[0]->shape());
        $this->assertEquals([5],$gradients[1]->shape());
        $this->assertEquals("[[7,7,7,7,7]]",$mo->toString($gradients[0]));
        $this->assertEquals("[2,2,2,2,2]",$mo->toString($gradients[1]));
    }
}
