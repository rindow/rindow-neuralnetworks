<?php
namespace RindowTest\NeuralNetworks\Gradient\Func\MaximumTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

class MaximumTest extends TestCase
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

    public function testSingleValue()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $a = $g->Variable($K->array([2.0,3.0,4.0,5.0]));
        $x = $g->Variable($K->array(4.0));
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$a,$x){
                $y = $g->maximum($a,$x);
                return $y;
            }
        );
        [$da,$dx] = $tape->gradient($y,[$a,$x]);
        $this->assertEquals("[4,4,4,5]",$mo->toString($y->value()));
        $this->assertEquals("[0,0,1,1]",$mo->toString($da));
        $this->assertEquals("2",$mo->toString($dx));
    }

    public function testSingleAndSinglesValue()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $a = $g->Variable($K->array(2.0));
        $x = $g->Variable($K->array(3.0));
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$a,$x){
                $y = $g->maximum($a,$x);
                return $y;
            }
        );
        [$da,$dx] = $tape->gradient($y,[$a,$x]);
        $this->assertEquals("3",$mo->toString($y->value()));
        $this->assertEquals("0",$mo->toString($da));
        $this->assertEquals("1",$mo->toString($dx));
    }

    public function testMatrixValue()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $a = $g->Variable($K->array([2.0,3.0,4.0,5.0]));
        $x = $g->Variable($K->array([4.0,4.0,4.0,4.0]));
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$a,$x) {
                $y = $g->maximum($a,$x);
                return $y;
            }
        );
        [$da,$dx] = $tape->gradient($y,[$a,$x]);

        $this->assertEquals("[4,4,4,5]",$mo->toString($y->value()));
        $this->assertEquals("[0,0,1,1]",$mo->toString($da));
        $this->assertEquals("[1,1,0,0]",$mo->toString($dx));
    }

    public function testMatrixBroadcast()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);

        $g = $nn->gradient();
        $a = $g->Variable($K->array([[2.0,3.0],[4.0,5.0]]));
        $x = $g->Variable($K->array([4.0,4.0]));
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$a,$x) {
                $y = $g->maximum($a,$x);
                return $y;
            }
        );
        [$da,$dx] = $tape->gradient($y,[$a,$x]);

        $this->assertEquals("[[4,4],[4,5]]",$mo->toString($y->value()));
        $this->assertEquals("[[0,0],[1,1]]",$mo->toString($da));
        $this->assertEquals("[1,1]",$mo->toString($dx));
    }
}
