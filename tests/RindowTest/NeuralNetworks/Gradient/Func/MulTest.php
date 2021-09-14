<?php
namespace RindowTest\NeuralNetworks\Gradient\Funct\MulTest;

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

        $x = $g->Variable($K->array(3.0));
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x){
                $y = $g->mul($x,$x);
                return $y;
            }
        );
        $this->assertEquals("9",$mo->toString($y->value()));
        $this->assertEquals("6",$mo->toString($tape->gradient($y,$x)));
    }

    public function testMatrixValue()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x = $g->Variable($K->array([3.0, 4.0]));
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x){
                $y = $g->mul($x,$x);
                return $y;
            }
        );

        $this->assertEquals("[9,16]",$mo->toString($y->value()));
        $this->assertEquals("[6,8]",$mo->toString($tape->gradient($y,$x)));
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
                $y = $g->mul($x0,$x1);
                return $y;
            }
        );

        $this->assertEquals("[[6,12],[12,20]]",$mo->toString($y->value()));
        $this->assertEquals("[6,8]",$mo->toString($tape->gradient($y,$x0)));
        $this->assertEquals("[[3,4],[3,4]]",$mo->toString($tape->gradient($y,$x1)));
    }
}
