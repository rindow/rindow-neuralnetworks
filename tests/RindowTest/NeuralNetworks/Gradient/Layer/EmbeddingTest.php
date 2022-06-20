<?php
namespace RindowTest\NeuralNetworks\Gradient\Layer\EmbeddingTest;

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
            [0,1,2],
            [2,1,0],
        ]));
        $layer = $nn->layers->Embedding($inputDim=3, $outputDim=4, input_length:3);

        $outputs = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$x) {
                $outputs = $layer($x,true);
                return $outputs;
            }
        );
        $gradients = $tape->gradient($outputs, $layer->weights());

        $this->assertEquals([2,3,4],$outputs->value()->shape());
        $this->assertCount(1,$layer->weights());
        $this->assertCount(1,$gradients);
        $this->assertEquals([3,4],$gradients[0]->shape());
    }
}
