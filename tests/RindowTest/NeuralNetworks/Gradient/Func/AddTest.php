<?php
namespace RindowTest\NeuralNetworks\Gradient\Func\AddTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

class AddTest extends TestCase
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
                $y = $g->add($x,$x);
                return $y;
            }
        );
        $this->assertEquals("6",$mo->toString($y->value()));
        $this->assertEquals("2",$mo->toString($tape->gradient($y,$x)));
    }

    public function testMatrixValue()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x = $g->Variable($K->array([3.0, 4.0]));
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x) {
                $y = $g->add($x,$x);
                return $y;
            }
        );

        $this->assertEquals("[6,8]",$mo->toString($y->value()));
        $this->assertEquals("[2,2]",$mo->toString($tape->gradient($y,$x)));
    }

    public function testMatrixBroadcast()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);

        $g = $nn->gradient();
        $x0 = $g->Variable($K->array([3.0, 4.0]));
        $x1 = $g->Variable($K->array([[2.0, 3.0],[4.0, 5.0]]));
        $y = $nn->with($tape=$g->GradientTape($persistent=true),
            function() use ($g,$x0,$x1) {
                $y = $g->add($x0,$x1);
                return $y;
            }
        );

        $this->assertEquals("[[5,7],[7,9]]",$mo->toString($y->value()));
        $this->assertEquals("[2,2]",$mo->toString($tape->gradient($y,$x0)));
        $this->assertEquals("[[1,1],[1,1]]",$mo->toString($tape->gradient($y,$x1)));
    }

    public function testScalarValue()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        // x + 1
        $x = $g->Variable($K->array([3.0, 4.0]));
        $c = $g->constant($K->array([8.0, 9.0]));
        [$y,$z] = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x,$c) {
                $y = $g->add($x,1);
                $z = $g->mul($y,$c);
                return [$y,$z];
            }
        );
        $this->assertEquals("[4,5]",$mo->toString($y->value()));
        $this->assertEquals("[8,9]",$mo->toString($tape->gradient($z,$x)));

        // 1 + x
        $x = $g->Variable($K->array([3.0, 4.0]));
        $c = $g->constant($K->array([8.0, 9.0]));
        [$y,$z] = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x,$c) {
                $y = $g->add(1,$x);
                $z = $g->mul($y,$c);
                return [$y,$z];
            }
        );
        $this->assertEquals("[4,5]",$mo->toString($y->value()));
        $this->assertEquals("[8,9]",$mo->toString($tape->gradient($z,$x)));
    }

    public function testTransposeBroadcast()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);

        $g = $nn->gradient();
        $x0 = $g->Variable($K->array([[2.0, 3.0, 4.0],[5.0, 6.0, 7.0]]));
        $x1 = $g->Variable($K->array([80.0, 90.0, 100.0]));
        $c = $g->constant($K->array([[1.0, 2.0, 3.0],[4.0, 5.0, 6.0]]));

        // No-Trans-Broadcast
        [$y,$z] = $nn->with($tape=$g->GradientTape($persistent=true),
            function() use ($g,$x0,$x1,$c) {
                $y = $g->add($x0,$x1);
                $z = $g->mul($y,$c);
                return [$y,$z];
            }
        );
        $this->assertEquals("[[82,93,104],[85,96,107]]",$mo->toString($y->value()));
        $this->assertEquals("[[1,2,3],[4,5,6]]",$mo->toString($tape->gradient($z,$x0)));
        $this->assertEquals("[5,7,9]",$mo->toString($tape->gradient($z,$x1)));


        $x0 = $g->Variable($K->array([[2.0, 3.0, 4.0],[5.0, 6.0, 7.0]]));
        $x1 = $g->Variable($K->array([80.0, 90.0]));
        $c = $g->constant($K->array([[1.0, 2.0, 3.0],[4.0, 5.0, 6.0]]));

        // Trans-Broadcast
        [$y,$z] = $nn->with($tape=$g->GradientTape($persistent=true),
            function() use ($g,$x0,$x1,$c) {
                $y = $g->add($x0,$x1,trans:true);
                $z = $g->mul($y,$c);
                return [$y,$z];
            }
        );
        $this->assertEquals("[[82,83,84],[95,96,97]]",$mo->toString($y->value()));
        $this->assertEquals("[[1,2,3],[4,5,6]]",$mo->toString($tape->gradient($z,$x0)));
        $this->assertEquals("[6,15]",$mo->toString($tape->gradient($z,$x1)));
    }

}
