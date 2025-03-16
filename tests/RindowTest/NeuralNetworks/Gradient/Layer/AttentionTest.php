<?php
namespace RindowTest\NeuralNetworks\Gradient\Layer\AttentionTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

class AttentionTest extends TestCase
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

        $query = $g->Variable($K->array([
            [0,1],
            [0,1],
        ],dtype:NDArray::int32));
        $value = $g->Variable($K->array([
            [0,1,2,3],
            [0,1,2,3],
        ],dtype:NDArray::int32));

        $embed1 = $nn->layers->Embedding($inputDim=2, $outputDim=3, input_length:2);
        $embed2 = $nn->layers->Embedding($inputDim=4, $outputDim=3, input_length:4);
        $layer = $nn->layers->Attention();

        $outputs = $nn->with($tape=$g->GradientTape(),
            function() use ($embed1,$embed2,$layer,$query,$value) {
                $q = $embed1($query);
                $v = $embed2($value);
                $outputs = $layer([$q,$v]);
                return $outputs;
            }
        );
        $gradients = $tape->gradient($outputs, [$query,$value]);

        $this->assertEquals([2,2,3],$outputs->value()->shape());
        $this->assertCount(0,$layer->weights());
        $this->assertCount(2,$gradients);
        $this->assertEquals([2,2],$gradients[0]->shape());
        $this->assertEquals([2,4],$gradients[1]->shape());
    }

    public function testWithReturnAttentionScores()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $query = $g->Variable($K->array([
            [0,1],
            [0,1],
        ],dtype:NDArray::int32));
        $value = $g->Variable($K->array([
            [0,1,2,3],
            [0,1,2,3],
        ],dtype:NDArray::int32));

        $embed1 = $nn->layers->Embedding($inputDim=2, $outputDim=3, input_length:2);
        $embed2 = $nn->layers->Embedding($inputDim=4, $outputDim=3, input_length:4);
        $layer = $nn->layers->Attention();

        $outputs = $nn->with($tape=$g->GradientTape(),
            function() use ($embed1,$embed2,$layer,$query,$value) {
                $q = $embed1($query);
                $v = $embed2($value);
                $outputs = $layer([$q,$v],returnAttentionScores:true);
                return $outputs;
            }
        );
        $this->assertCount(2,$outputs);
        $gradients = $tape->gradient($outputs[0], [$query,$value]);

        $this->assertEquals([2,2,3],$outputs[0]->value()->shape());
        $this->assertCount(0,$layer->weights());
        $this->assertCount(2,$gradients);
        $this->assertEquals([2,2],$gradients[0]->shape());
        $this->assertEquals([2,4],$gradients[1]->shape());
    }
}
