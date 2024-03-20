<?php
namespace RindowTest\NeuralNetworks\Gradient\Func\NotEqualTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;

class NotEqualTest extends TestCase
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
        $y = $g->Variable($K->array(2));
        $z = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x,$y){
                $z = $g->notEqual($x,$y);
                return $z;
            }
        );
        $this->assertEquals("1",$mo->toString($z->value()));
        $this->assertTrue($x->isbackpropagatable());
        $this->assertTrue($y->isbackpropagatable());
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
        $y = $g->Variable($K->array([3.0, 2.0]));
        $z = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x,$y) {
                $y = $g->notEqual($x,$y);
                return $y;
            }
        );

        $this->assertEquals("[0,1]",$mo->toString($z->value()));
        $this->assertTrue($x->isbackpropagatable());
        $this->assertTrue($y->isbackpropagatable());
        $this->assertFalse($z->isbackpropagatable());
        try {
            $tape->gradient($z,$x);
        } catch(\Throwable $e) {
            $error = $e->getMessage();
        }
        $this->assertStringStartsWith("No applicable gradient found for source",$error);
    }

    public function testIntegerValue()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x = $g->Variable($K->array([3.0, 4.0]),NDArray::int8);
        $y = $g->Variable($K->array([3.0, 2.0]),NDArray::int8);
        $z = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x,$y) {
                $y = $g->notEqual($x,$y);
                return $y;
            }
        );

        $this->assertEquals("[0,1]",$mo->toString($z->value()));
        $this->assertTrue($x->isbackpropagatable());
        $this->assertTrue($y->isbackpropagatable());
        $this->assertFalse($z->isbackpropagatable());
        try {
            $tape->gradient($z,$x);
        } catch(\Throwable $e) {
            $error = $e->getMessage();
        }
        $this->assertStringStartsWith("No applicable gradient found for source",$error);
    }
}
