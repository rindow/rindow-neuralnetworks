<?php
namespace RindowTest\NeuralNetworks\Gradient\Core\FunctionTest;

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

    public function testVariableNormal()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);

        $a = $nn->gradient->Variable($K->array(1));
        $this->assertEquals("1",$mo->toString($a->value()));
    }

    public function testFunctionNormal()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x = $nn->gradient->Variable($K->array(10));
        $y = $g->square($x);
        $this->assertEquals("100",$mo->toString($y->value()));
    }

    public function testChainFunctionNormal()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x = $g->Variable($K->array(0.5));
        $a = $g->square($x);
        $b = $g->exp($a);
        $y = $g->square($b);

        $this->assertStringStartsWith("1.64872",$mo->toString($y->value()));
    }

    public function testGradientFunctionNormal()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x = $g->Variable($K->array(0.5));
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x){
                $y = $g->square($g->exp($g->square($x)));
                return $y;
            }
        );

        $this->assertStringStartsWith("1.64872",$mo->toString($y->value()));
        $this->assertStringStartsWith("3.29744",$mo->toString($tape->gradient($y,$x)));
    }

    public function testGradientFunctionWithTwoArgs1()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $a = $g->Variable($K->array(2.0));
        $b = $g->Variable($K->array(3.0));
        $c = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$a,$b){
                $c = $g->add($g->square($a),$g->square($b));
                return $c;
            }
        );

        $this->assertEquals("13",$mo->toString($c->value()));
        $this->assertEquals("4",$mo->toString($tape->gradient($c,$a)));
        $this->assertEquals("6",$mo->toString($tape->gradient($c,$b)));
    }

    public function testGradientFunctionWithTwoArgs2()
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

    public function testGradientFunctionWithTwoArgs3()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x = $g->Variable($K->array(3.0));
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x){
                $y = $g->add($g->add($x,$x),$x);
                return $y;
            }
        );

        $this->assertEquals("9",$mo->toString($y->value()));
        $this->assertEquals("3",$mo->toString($tape->gradient($y,$x)));
    }

    public function testGradientFunctionWithTwoArgs4()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $a = $g->Variable($K->array(2.0));
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$a){
                $b = $g->square($a);
                $y = $g->add($g->square($b),$g->square($b));
                return $y;
            }
        );

        $this->assertEquals("32",$mo->toString($y->value()));
        $this->assertEquals("64",$mo->toString($tape->gradient($y,$a)));
    }

    public function testGradientWithConfig()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);

        $g = $nn->gradient();
        $a = $g->Variable($K->array(2.0));
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$a){
                $b = $g->square($a);
                $y = $g->add($g->square($b),$g->square($b));
                return $y;
            }
        );

        $this->assertEquals("32",$mo->toString($y->value()));
        $this->assertEquals("64",$mo->toString($tape->gradient($y,$a)));

        // No Grad
        $x = $g->Variable($K->array(2.0));
        $y = $g->square($x);
        $this->assertEquals("4",$mo->toString($y->value()));
    }

    public function testGradientMultiParams()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x1 = $g->Variable($K->array(2.0));
        $x2 = $g->Variable($K->array(3.0));
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x1,$x2){
                $y = $g->mul($x1,$x2);
                return $y;
            }
        );

        $grads = $tape->gradient($y,[$x1,$x2]);
        $this->assertCount(2,$grads);
        $this->assertStringStartsWith("6",$mo->toString($y->value()));
        $this->assertStringStartsWith("3",$mo->toString($grads[0]));
        $this->assertStringStartsWith("2",$mo->toString($grads[1]));
    }
}
