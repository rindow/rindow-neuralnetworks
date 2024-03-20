<?php
namespace RindowTest\NeuralNetworks\Gradient\Layer\AveragePooling1DTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Layer\AveragePooling1D;
use InvalidArgumentException;

class AveragePooling1DTest extends TestCase
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

    public function testNormalForwardAndBackward()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x = $g->Variable($K->ones([2,4,3]));
        $layer = $nn->layers->AveragePooling1D(input_shape:[4,3]);

        $outputs = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$x) {
                $outputs = $layer($x,true);
                return $outputs;
            }
        );
        $this->assertCount(0,$layer->weights());
        $gradients = $tape->gradient($outputs, $x);

        $this->assertEquals([2,4,3],$gradients->shape());
        $this->assertEquals(
            "[[[1,1,1],[1,1,1]],[[1,1,1],[1,1,1]]]",
            $mo->toString($outputs->value()));
        $this->assertEquals(
            "[[[0.5,0.5,0.5],[0.5,0.5,0.5],[0.5,0.5,0.5],[0.5,0.5,0.5]],".
             "[[0.5,0.5,0.5],[0.5,0.5,0.5],[0.5,0.5,0.5],[0.5,0.5,0.5]]]",
            $mo->toString($gradients));
    }
}
