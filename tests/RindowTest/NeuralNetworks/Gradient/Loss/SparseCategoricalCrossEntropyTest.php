<?php
namespace RindowTest\NeuralNetworks\Gradient\Loss\SparseCategoricalCrossEntropyTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

class SparseCategoricalCrossEntropyTest extends TestCase
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

    public function testDefault()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x = $g->Variable($K->array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]]));
        $t = $g->Variable($K->array([1, 2],NDArray::int32));
        $activation = $nn->layers->Activation('softmax',input_shape:[3]);
        $loss = $nn->losses->SparseCategoricalCrossEntropy();

        $func = $g->Function(
            function($t,$x) use ($g,$activation,$loss) {
                $xx = $activation($x,true);
                $outputs = $loss($t,$xx);
                $outputs = $g->mul($outputs,$g->Variable(2));
                return $outputs;
            }
        );

        // build
        $outputs = $nn->with($tape=$g->GradientTape(),$func,[$t,$x],true);
        $gradients = $tape->gradient($outputs, $x);
        //$optimizer->update($model->params(),$gradients);
        $this->assertStringStartsWith((string)(0.98689*2),$mo->toString($outputs->value()));
        $this->assertEquals([2,3],$gradients->shape());
        $this->assertTrue($mo->la()->isclose(
            $mo->array([[ 0.22671451, -0.44237202,  0.21565753],
                        [ 0.24914327,  0.50171292, -0.75085628]]),
            $K->ndarray($gradients)));

        // exec
        $outputs = $nn->with($tape=$g->GradientTape(),$func,[$t,$x],true);
        $gradients = $tape->gradient($outputs, $x);
        //$optimizer->update($model->params(),$gradients);
        $this->assertStringStartsWith((string)(0.98689*2),$mo->toString($outputs->value()));
        $this->assertEquals([2,3],$gradients->shape());
        $this->assertTrue($mo->la()->isclose(
            $mo->array([[ 0.22671451, -0.44237202,  0.21565753],
                        [ 0.24914327,  0.50171292, -0.75085628]]),
            $K->ndarray($gradients)));
    }

    public function testFromLogits()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x = $g->Variable($K->array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]]));
        $t = $g->Variable($K->array([1, 2],NDArray::int32));
        $loss = $nn->losses->SparseCategoricalCrossEntropy(
            from_logits:true
        );

        $func = $g->Function(
            function($t,$x) use ($g,$loss) {
                $outputs = $loss($t,$x);
                $outputs = $g->mul($outputs,$g->Variable(2));
                return $outputs;
            }
        );

        // build
        $outputs = $nn->with($tape=$g->GradientTape(),$func,[$t,$x],true);
        $gradients = $tape->gradient($outputs, $x);
        //$optimizer->update($model->params(),$gradients);
        $this->assertStringStartsWith((string)(0.98689*2),$mo->toString($outputs->value()));
        $this->assertEquals([2,3],$gradients->shape());
        $this->assertTrue($mo->la()->isclose(
            $mo->array([[ 0.22671457, -0.44237214,  0.21565758],
                        [ 0.24914339,  0.50171316, -0.75085664 ]]),
            $K->ndarray($gradients)));

        // exec
        $outputs = $nn->with($tape=$g->GradientTape(),$func,[$t,$x],true);
        $gradients = $tape->gradient($outputs, $x);
        //$optimizer->update($model->params(),$gradients);
        $this->assertStringStartsWith((string)(0.98689*2),$mo->toString($outputs->value()));
        $this->assertEquals([2,3],$gradients->shape());
        $this->assertTrue($mo->la()->isclose(
            $mo->array([[ 0.22671457, -0.44237214,  0.21565758],
                        [ 0.24914339,  0.50171316, -0.75085664 ]]),
            $K->ndarray($gradients)));
    }
}
