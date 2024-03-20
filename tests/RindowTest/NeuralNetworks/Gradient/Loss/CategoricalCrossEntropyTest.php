<?php
namespace RindowTest\NeuralNetworks\Gradient\Loss\CategoricalCrossEntropyTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

class CategoricalCrossEntropyTest extends TestCase
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
        $t = $K->array([[0, 1, 0], [0, 0, 1]]);
        $activation = $nn->layers->Activation('softmax',input_shape:[3]);
        $loss = $nn->losses->CategoricalCrossEntropy();

        $outputs = $nn->with($tape=$g->GradientTape(),
            function() use ($activation,$loss,$t,$x) {
                $xx = $activation($x,true);
                $outputs = $loss($t,$xx);
                return $outputs;
            }
        );
        $gradients = $tape->gradient($outputs, $x);
        //$optimizer->update($model->params(),$gradients);
        $this->assertStringStartsWith("0.98689",$mo->toString($outputs->value()));
        $this->assertEquals([2,3],$gradients->shape());
        $this->assertTrue($mo->la()->isclose(
            $mo->array([[ 0.11335728, -0.22118606,  0.10782879],
                        [ 0.12457169,  0.25085658, -0.3754283 ]]),
            $K->ndarray($gradients)));
    }

    public function testFromLogits()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x = $g->Variable($K->array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]]));
        $t = $K->array([[0, 1, 0], [0, 0, 1]]);
        $loss = $nn->losses->CategoricalCrossEntropy(
            from_logits:true
        );

        $outputs = $nn->with($tape=$g->GradientTape(),
            function() use ($loss,$t,$x) {
                $outputs = $loss($t,$x);
                return $outputs;
            }
        );
        $gradients = $tape->gradient($outputs, $x);
        //$optimizer->update($model->params(),$gradients);
        $this->assertStringStartsWith("0.98689",$mo->toString($outputs->value()));
        $this->assertEquals([2,3],$gradients->shape());
        $this->assertTrue($mo->la()->isclose(
            $mo->array([[ 0.11335728, -0.22118606,  0.10782879],
                        [ 0.12457169,  0.25085658, -0.3754283 ]]),
            $K->ndarray($gradients)));
    }
}
