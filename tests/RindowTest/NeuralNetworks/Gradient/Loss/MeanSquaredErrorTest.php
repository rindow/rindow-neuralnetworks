<?php
namespace RindowTest\NeuralNetworks\Gradient\Loss\MeanSquaredErrorTest;

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

        $x = $g->Variable($K->array([[0., 1.], [0., 0.]]));
        $t = $K->array([[1., 1.], [1., 0.]]);
        $loss = $nn->losses->MeanSquaredError();

        $outputs = $nn->with($tape=$g->GradientTape(),
            function() use ($loss,$t,$x) {
                $outputs = $loss($t,$x);
                return $outputs;
            }
        );
        $gradients = $tape->gradient($outputs, $x);
        //$optimizer->update($model->params(),$gradients);
        $this->assertStringStartsWith("0.5",$mo->toString($outputs->value()));
        $this->assertEquals([2,2],$gradients->shape());
        $this->assertTrue($mo->la()->isclose(
            $mo->array([[ -0.5,  0.0],
                        [ -0.5,  0.0]]),
            $K->ndarray($gradients)));
    }
}
