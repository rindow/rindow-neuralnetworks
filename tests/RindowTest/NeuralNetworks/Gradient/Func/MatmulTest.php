<?php
namespace RindowTest\NeuralNetworks\Gradient\Func\MatmulTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
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

    public function testMatrixValueNormal()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $a = $g->Variable($K->array([[1,2,3],[4,5,6]]));
        $b = $g->Variable($K->array([[7,8],[9,10],[11,12]]));
        $c = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$a,$b) {
                $c = $g->matmul($a,$b);
                return $c;
            }
        );
        $gradients = $tape->gradient($c,[$a,$b]);

        $this->assertEquals("[[58,64],[139,154]]",$mo->toString($c->value()));
        $this->assertEquals("[[15,19,23],[15,19,23]]",$mo->toString($gradients[0]));
        $this->assertEquals("[[5,5],[7,7],[9,9]]",$mo->toString($gradients[1]));
    }

    public function testMatrixValueTransposeA()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $a = $g->Variable($K->array([[1,4],[2,5],[3,6]]));
        $b = $g->Variable($K->array([[7,8],[9,10],[11,12]]));
        $c = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$a,$b) {
                $c = $g->matmul($a,$b,transpose_a:true);
                return $c;
            }
        );
        $gradients = $tape->gradient($c,[$a,$b]);

        $this->assertEquals("[[58,64],[139,154]]",$mo->toString($c->value()));
        $this->assertEquals("[[15,15],[19,19],[23,23]]",$mo->toString($gradients[0]));
        $this->assertEquals("[[5,5],[7,7],[9,9]]",$mo->toString($gradients[1]));
    }

    public function testMatrixValueTransposeB()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $a = $g->Variable($K->array([[1,2,3],[4,5,6]]));
        $b = $g->Variable($K->array([[7,9,11],[8,10,12]]));
        $c = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$a,$b) {
                $c = $g->matmul($a,$b,transpose_b:true);
                return $c;
            }
        );
        $gradients = $tape->gradient($c,[$a,$b]);

        $this->assertEquals("[[58,64],[139,154]]",$mo->toString($c->value()));
        $this->assertEquals("[[15,19,23],[15,19,23]]",$mo->toString($gradients[0]));
        $this->assertEquals("[[5,7,9],[5,7,9]]",$mo->toString($gradients[1]));
    }

    public function testMatrixValueTransposeAandB()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $a = $g->Variable($K->array([[1,2,3],[4,5,6]]));
        $b = $g->Variable($K->array([[7,8],[9,10],[11,12]]));
        $c = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$a,$b) {
                $c = $g->matmul($a,$b, transpose_a:true, transpose_b:true);
                return $c;
            }
        );
        $gradients = $tape->gradient($c,[$a,$b]);

        $this->assertEquals("[[39,49,59],[54,68,82],[69,87,105]]",$mo->toString($c->value()));
        $this->assertEquals("[[27,27,27],[30,30,30]]",$mo->toString($gradients[0]));
        $this->assertEquals("[[6,15],[6,15],[6,15]]",$mo->toString($gradients[1]));
    }

    public function testBatchStyleNormal()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $a = $g->Variable($K->array($mo->arange(12,1,null,NDArray::float32)->reshape([2,2,3])));
        $b = $g->Variable($K->array($mo->arange(12,13,null,NDArray::float32)->reshape([2,3,2])));
        $c = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$a,$b) {
                $c = $g->matmul($a,$b);
                return $c;
            }
        );
        $this->assertEquals("[[[94,100],[229,244]],[[508,532],[697,730]]]",$mo->toString($c->value()));
        $gradients = $tape->gradient($c,[$a,$b]);

        $this->assertEquals("[[[27,31,35],[27,31,35]],[[39,43,47],[39,43,47]]]",$mo->toString($gradients[0]));
        $this->assertEquals("[[[5,5],[7,7],[9,9]],[[17,17],[19,19],[21,21]]]",$mo->toString($gradients[1]));
    }

    public function testBatchStyleTransposeA()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $a = $g->Variable($K->array($mo->arange(12,1,null,NDArray::float32)->reshape([2,3,2])));
        $b = $g->Variable($K->array($mo->arange(12,13,null,NDArray::float32)->reshape([2,3,2])));
        $c = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$a,$b) {
                $c = $g->matmul($a,$b, transpose_a: true);
                return $c;
            }
        );
        $this->assertEquals("[[[143,152],[188,200]],[[575,602],[638,668]]]",$mo->toString($c->value()));
        $gradients = $tape->gradient($c,[$a,$b]);

        $this->assertEquals("[[[27,27],[31,31],[35,35]],[[39,39],[43,43],[47,47]]]",$mo->toString($gradients[0]));
        $this->assertEquals("[[[3,3],[7,7],[11,11]],[[15,15],[19,19],[23,23]]]",$mo->toString($gradients[1]));
    }

    public function testBatchStyleTransposeB()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $a = $g->Variable($K->array($mo->arange(12,1,null,NDArray::float32)->reshape([2,2,3])));
        $b = $g->Variable($K->array($mo->arange(12,13,null,NDArray::float32)->reshape([2,2,3])));
        $c = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$a,$b) {
                $c = $g->matmul($a,$b, transpose_b:true);
                return $c;
            }
        );
        $this->assertEquals("[[[86,104],[212,257]],[[482,554],[662,761]]]",$mo->toString($c->value()));
        $gradients = $tape->gradient($c,[$a,$b]);

        $this->assertEquals("[[[29,31,33],[29,31,33]],[[41,43,45],[41,43,45]]]",$mo->toString($gradients[0]));
        $this->assertEquals("[[[5,7,9],[5,7,9]],[[17,19,21],[17,19,21]]]",$mo->toString($gradients[1]));
    }
}
