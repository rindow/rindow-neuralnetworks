<?php
namespace RindowTest\NeuralNetworks\Gradient\Layer\DenseTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

class DenseTest extends TestCase
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
        $layer = $nn->layers->Dense($units=5,input_shape:[1]);

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

        //var_dump($K->toString($outputs->value()));
        //var_dump(array_map(function($x)use($K){return $K->toString($x);},$layer->getParams()));
        //var_dump(array_map(function($x)use($K){return $K->toString($x);},$layer->getGrads()));
    }

    public function testChain()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x = $g->Variable($K->array([[3.0], [4.0]]));
        $layer0 = $nn->layers->Dense($units=5,input_shape:[1]);
        $layer1 = $nn->layers->Dense($units=5);

        $outputs = $nn->with($tape=$g->GradientTape(),
            function() use ($layer0,$layer1,$x) {
                $outputs = $layer0($x,true);
                $outputs = $layer1($outputs,true);
                return $outputs;
            }
        );
        $this->assertEquals([2,5],$outputs->shape());

        $gradients = $tape->gradient($outputs, array_merge($layer0->weights(),$layer1->weights()));
        //$optimizer->update($model->params(),$gradients);

        $this->assertCount(4,$gradients);
        $this->assertEquals([1,5],$gradients[0]->shape());
        $this->assertEquals([5],$gradients[1]->shape());
        $this->assertEquals([5,5],$gradients[2]->shape());
        $this->assertEquals([5],$gradients[3]->shape());

        $this->assertTrue(true);
    }

    public function testNoGradient()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x = $g->Variable($K->array([[3.0], [4.0]]));
        $layer0 = $nn->layers->Dense($units=5,input_shape:[1]);
        $layer1 = $nn->layers->Dense($units=5);

        $outputs = $layer0($x,true);
        $outputs = $layer1($outputs,true);

        $this->assertEquals([2,5],$outputs->shape());
        $this->assertTrue(true);

    }
}
