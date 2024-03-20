<?php
namespace RindowTest\NeuralNetworks\Gradient\Layer\ActivationTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

class ActivationTest extends TestCase
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

        $x = $g->Variable($K->array([
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
        ]));
        $layer = $nn->layers->Activation('relu', input_shape:[5]);

        $outputs = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$x) {
                $outputs = $layer($x,true);
                return $outputs;
            }
        );
        $gradients = $tape->gradient($outputs, $x);
        $this->assertEquals([4,5],$gradients->shape());
        $this->assertEquals("[[0,0,0,1,1],[0,0,0,1,1],[0,0,0,1,1],[0,0,0,1,1]]",
            $mo->toString($gradients));
    }
}
