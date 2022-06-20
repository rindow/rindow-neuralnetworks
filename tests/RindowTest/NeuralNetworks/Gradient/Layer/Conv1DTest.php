<?php
namespace RindowTest\NeuralNetworks\Gradient\Layer\Conv1DTest;

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
        $layer = $nn->layers->Conv1D(
            $filters=2,
            $kernel_size=2,
            ['input_shape'=>[3,1]]);

        $outputs = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$x) {
                $outputs = $layer($x,true);
                return $outputs;
            }
        );
        $gradients = $tape->gradient($outputs, $layer->weights());

        $this->assertCount(2,$layer->weights());
        $this->assertCount(2,$gradients);
        $this->assertEquals([2,1,2],$gradients[0]->shape());
        $this->assertEquals([2],$gradients[1]->shape());
    }
}
