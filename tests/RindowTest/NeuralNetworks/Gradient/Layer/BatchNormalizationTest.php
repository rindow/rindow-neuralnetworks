<?php
namespace RindowTest\NeuralNetworks\Gradient\Layer\BatchNormalizationTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use InvalidArgumentException;

class BatchNormalizationTest extends TestCase
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
        $flatten = $nn->layers->Flatten(input_shape:[3,1]);
        $layer =   $nn->layers->BatchNormalization();

        $outputs = $nn->with($tape=$g->GradientTape(),
            function() use ($flatten,$layer,$x) {
                $x1 = $flatten($x,true);
                $outputs = $layer($x1,true);
                return $outputs;
            }
        );
        $gradients = $tape->gradient($outputs, $layer->trainableVariables());

        $this->assertCount(4,$layer->variables());
        $this->assertCount(2,$layer->trainableVariables());
        $this->assertCount(2,$gradients);
        $this->assertEquals([2,3],$outputs->value()->shape());
        $this->assertEquals([3],$gradients[0]->shape());
        $this->assertEquals([3],$gradients[1]->shape());
        $this->assertEquals("[[0,0,0],[0,0,0]]",$mo->toString($outputs->value()));
        $this->assertEquals("[2,2,2]",$mo->toString($gradients[0]));
        $this->assertEquals("[0,0,0]",$mo->toString($gradients[1]));
    }
}
