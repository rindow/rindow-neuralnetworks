<?php
namespace RindowTest\NeuralNetworks\Gradient\Layer\FlattenTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use InvalidArgumentException;

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

    public function testNormalForwardAndBackward()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x = $g->Variable($K->array([
            [[0.0],[0.0],[6.0]],
            [[0.0],[0.0],[6.0]],
        ]));
        $layer1 = $nn->layers->Flatten(input_shape:[3,1]);
        $layer2 = $nn->layers->Flatten();

        $outputs = $nn->with($tape=$g->GradientTape(),
            function() use ($layer1,$layer2,$x) {
                $x1 = $layer1($x,true);
                $outputs = $layer2($x1,true);
                return $outputs;
            }
        );
        $gradients = $tape->gradient($outputs, $x);

        $this->assertCount(0,$layer1->weights());
        $this->assertCount(0,$layer2->weights());
        $this->assertEquals([2,3],$outputs->value()->shape());
        $this->assertEquals([2,3,1],$gradients->shape());
        $this->assertEquals("[[0,0,6],[0,0,6]]",$mo->toString($outputs->value()));
        $this->assertEquals("[[[1],[1],[1]],[[1],[1],[1]]]",$mo->toString($gradients));
    }
}
