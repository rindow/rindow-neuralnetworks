<?php
namespace RindowTest\NeuralNetworks\Gradient\Layer\ConcatenateTest;

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

    public function testNormal()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x = $K->array([
            [0,1,2,3],
            [4,5,6,7],
        ]);
        $this->assertEquals([2,4],$x->shape());
        $x = $g->Variable($x);
        $flatten1 = $nn->layers->Flatten(
            input_shape:[4]);
        $flatten2 = $nn->layers->Flatten(
            input_shape:[4]);
        $layer = $nn->layers->Concatenate();

        $outputs = $nn->with($tape=$g->GradientTape(),
            function() use ($flatten1,$flatten2,$layer,$x) {
                $outputs = $layer([$flatten1($x,true),$flatten2($x,true)],true);
                return $outputs;
            }
        );
        $this->assertEquals([2,8],$outputs->value()->shape());
        $gradients = $tape->gradient($outputs, $x);

        $this->assertCount(0,$layer->weights());
        $this->assertEquals([2,4],$gradients->shape());
    }
}
