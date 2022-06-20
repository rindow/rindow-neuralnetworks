<?php
namespace RindowTest\NeuralNetworks\Gradient\Loss\BinaryCrossEntropyTest;

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

    public function testDefault()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x = $g->Variable($K->array([-18.6, 0.51, 2.94, -12.8]));
        $t = $K->array([0, 1, 0, 0]);
        $activation = $nn->layers->Activation('sigmoid',input_shape:[]);
        $loss = $nn->losses->BinaryCrossEntropy();

        $outputs = $nn->with($tape=$g->GradientTape(),
            function() use ($activation,$loss,$t,$x) {
                $xx = $activation($x,true);
                $outputs = $loss($t,$xx);
                return $outputs;
            }
        );
        $gradients = $tape->gradient($outputs, $x);
        //$optimizer->update($model->params(),$gradients);
        $this->assertStringStartsWith("0.86545",$mo->toString($outputs->value()));
        $this->assertEquals([4],$gradients->shape());
        $this->assertTrue($mo->la()->isclose(
            $mo->array([ 2.0895967e-09, -9.3798377e-02, 2.3744719e-01, 6.9019114e-07]),
            $K->ndarray($gradients)));
    }

    public function testFromLogits()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x = $g->Variable($K->array([-18.6, 0.51, 2.94, -12.8]));
        $t = $K->array([0, 1, 0, 0]);
        $loss = $nn->losses->BinaryCrossEntropy(
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
        $this->assertStringStartsWith("0.86545",$mo->toString($outputs->value()));
        $this->assertEquals([4],$gradients->shape());
        $this->assertTrue($mo->la()->isclose(
            $mo->array([ 2.0895967e-09, -9.3798377e-02, 2.3744719e-01, 6.9019114e-07]),
            $K->ndarray($gradients)));
    }
}
