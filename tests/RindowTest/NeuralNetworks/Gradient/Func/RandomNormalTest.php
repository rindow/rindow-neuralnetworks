<?php
namespace RindowTest\NeuralNetworks\Gradient\Func\RandomNormalTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;

class RandomNormalTest extends TestCase
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
                $z = $g->randomNormal($x);
                return $z;
            }
        );
        $this->assertEquals([],$z->shape());
        $this->assertEquals($x->dtype(),$z->dtype());
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
                $y = $g->randomNormal($x);
                return $y;
            }
        );

        $this->assertEquals([2],$z->shape());
        $this->assertEquals($x->dtype(),$z->dtype());
        $this->assertTrue($x->isbackpropagatable());
        $this->assertFalse($z->isbackpropagatable());
        try {
            $tape->gradient($z,$x);
        } catch(\Throwable $e) {
            $error = $e->getMessage();
        }
        $this->assertStringStartsWith("No applicable gradient found for source",$error);
    }

    public function testMuiltSamples()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x = $g->Variable($K->array([3.0, 4.0]));
        $z = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x) {
                $y = $g->randomNormal($x,batchShape:[3]);
                return $y;
            }
        );

        $this->assertEquals([3,2],$z->shape());
        $this->assertEquals($x->dtype(),$z->dtype());
        $this->assertTrue($x->isbackpropagatable());
        $this->assertFalse($z->isbackpropagatable());
        try {
            $tape->gradient($z,$x);
        } catch(\Throwable $e) {
            $error = $e->getMessage();
        }
        $this->assertStringStartsWith("No applicable gradient found for source",$error);
    }

    public function testWithOptionValues()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x = $g->Variable($K->array([3.0, 4.0]));
        $z = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x) {
                $y = $g->randomNormal($x,mean:2.0,scale:0.5,seed:120);
                return $y;
            }
        );

        $this->assertEquals([2],$z->shape());
        $this->assertEquals($x->dtype(),$z->dtype());
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
