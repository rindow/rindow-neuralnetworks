<?php
namespace RindowTest\NeuralNetworks\Gradient\Func\ZerosLikeTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;

class ZerosLikeTest extends TestCase
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

        $x = $g->Variable($K->array(1));
        $z = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x){
                $z = $g->zerosLike($x);
                return $z;
            }
        );
        $this->assertEquals("0",$mo->toString($z->value()));
        $this->assertTrue($x->isbackpropagatable());
        $this->assertFalse($z->isbackpropagatable());
        try {
            $tape->gradient($z,$x);
        } catch(\Throwable $e) {
            $error = $e->getMessage();
        }
        $this->assertStringStartsWith("No applicable gradient found for source",$error);
    }

    public function testMatrixValue()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x = $g->Variable($K->array([3.0, 4.0]));
        $z = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x) {
                $y = $g->zerosLike($x);
                return $y;
            }
        );

        $this->assertEquals("[0,0]",$mo->toString($z->value()));
        $this->assertTrue($x->isbackpropagatable());
        $this->assertFalse($z->isbackpropagatable());
        try {
            $tape->gradient($z,$x);
        } catch(\Throwable $e) {
            $error = $e->getMessage();
        }
        $this->assertStringStartsWith("No applicable gradient found for source",$error);
    }
}
