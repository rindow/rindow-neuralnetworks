<?php
namespace RindowTest\NeuralNetworks\Gradient\Func\BandpartTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;

class BandpartTest extends TestCase
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

        $x = $g->Variable($K->ones([2,3,3]));
        $orgx = $K->copy($x->value());
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x){
                $y = $g->bandpart($x,0,-1);
                return $y;
            }
        );
        $this->assertEquals($x->toArray(),$orgx->toArray());
        $this->assertTrue($x->isbackpropagatable());
        $this->assertFalse($y->isbackpropagatable());
        $this->assertEquals([
            [[1,1,1],
             [0,1,1],
             [0,0,1]],
            [[1,1,1],
             [0,1,1],
             [0,0,1]],
        ],$y->toArray());
        try {
            $tape->gradient($y,$x);
        } catch(\Throwable $e) {
            $error = $e->getMessage();
        }
        $this->assertStringStartsWith("No applicable gradient found for source",$error);
    }
}
