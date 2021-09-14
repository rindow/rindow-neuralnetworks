<?php
namespace RindowTest\NeuralNetworks\Gradient\Func\SubTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

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

    public function testSingleValue()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x0 = $g->Variable($K->array(2.0));
        $x1 = $g->Variable($K->array(3.0));
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x0,$x1){
                $y = $g->sub($x0,$x1);
                return $y;
            }
        );

        $this->assertEquals("-1",$mo->toString($y->value()));
        $this->assertEquals("1",$mo->toString($tape->gradient($y,$x0)));
        $this->assertEquals("-1",$mo->toString($tape->gradient($y,$x1)));
    }

    public function testMatrixValue()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);

        $g = $nn->gradient();
        $x0 = $g->Variable($K->array([1.0, 3.0]));
        $x1 = $g->Variable($K->array([3.0, 4.0]));
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x0,$x1){
                $y = $g->sub($x0,$x1);
                return $y;
            }
        );

        $this->assertEquals("[-2,-1]",$mo->toString($y->value()));
        $this->assertEquals("[1,1]",$mo->toString($tape->gradient($y,$x0)));
        $this->assertEquals("[-1,-1]",$mo->toString($tape->gradient($y,$x1)));
    }

    public function testMatrixBroadcast()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);

        $g = $nn->gradient();
        $x0 = $g->Variable($K->array([3.0, 4.0]));
        $x1 = $g->Variable($K->array([[2.0, 3.0],[4.0, 5.0]]));
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x0,$x1){
                $y = $g->sub($x0,$x1);
                return $y;
            }
        );

        $this->assertEquals("[[1,1],[-1,-1]]",$mo->toString($y->value()));
        $this->assertEquals("[2,2]",$mo->toString($tape->gradient($y,$x0)));
        $this->assertEquals("[[-1,-1],[-1,-1]]",$mo->toString($tape->gradient($y,$x1)));
    }
}
