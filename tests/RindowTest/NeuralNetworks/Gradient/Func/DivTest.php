<?php
namespace RindowTest\NeuralNetworks\Gradient\Func\DivTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

class DivTest extends TestCase
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

        $x0 = $g->Variable($K->array(3.0));
        $x1 = $g->Variable($K->array(2.0));
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x0,$x1){
                $y = $g->div($x0,$x1);
                return $y;
            }
        );

        $this->assertEquals("1.5",$mo->toString($y->value()));
        $this->assertEquals("0.5",$mo->toString($tape->gradient($y,$x0)));
        $this->assertEquals("-0.75",$mo->toString($tape->gradient($y,$x1)));
    }

    public function testMatrixValue()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x0 = $g->Variable($K->array([1.0, 3.0]));
        $x1 = $g->Variable($K->array([4.0, 2.0]));
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x0,$x1){
                $y = $g->div($x0,$x1);
                return $y;
            }
        );

        $this->assertEquals("[0.25,1.5]",$mo->toString($y->value()));
        $this->assertEquals("[0.25,0.5]",$mo->toString($tape->gradient($y,$x0)));
        $this->assertEquals("[-0.0625,-0.75]",$mo->toString($tape->gradient($y,$x1)));
    }

    public function testMatrixBroadcast()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x0 = $g->Variable($K->array([1.0, 0.5]));
        $x1 = $g->Variable($K->array([[4.0, 2.0],[2.0, 4.0]]));
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x0,$x1){
                $y = $g->div($x0,$x1);
                return $y;
            }
        );

        $this->assertEquals("[[0.25,0.25],[0.5,0.125]]",$mo->toString($y->value()));
        $this->assertEquals("[0.75,0.75]",$mo->toString($tape->gradient($y,$x0)));
        $this->assertEquals("[[-0.0625,-0.125],[-0.25,-0.03125]]",$mo->toString($tape->gradient($y,$x1)));
    }

    public function testScalarValue()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        // x / 2
        $x = $g->Variable($K->array([4.0, 8.0]));
        $c = $g->constant($K->array([8.0, 9.0]));
        [$y,$z] = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x,$c) {
                $y = $g->div($x,2);
                $z = $g->mul($y,$c);
                return [$y,$z];
            }
        );
        $this->assertEquals("[2,4]",$mo->toString($y->value()));
        $this->assertEquals("[16,18]",$mo->toString($tape->gradient($z,$x)));

        // 8 / x
        $x = $g->Variable($K->array([2.0, 4.0]));
        $c = $g->constant($K->array([8.0, 9.0]));
        [$y,$z] = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x,$c) {
                $y = $g->div(8,$x);
                $z = $g->mul($y,$c);
                return [$y,$z];
            }
        );
        $this->assertEquals("[4,2]",$mo->toString($y->value()));
        $this->assertEquals("[-16,-4.5]",$mo->toString($tape->gradient($z,$x)));
    }

}
