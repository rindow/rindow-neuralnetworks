<?php
namespace RindowTest\NeuralNetworks\Gradient\Func\ScaleTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

class ScaleTest extends TestCase
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

    public function testScalar()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $func = $g->Function(
            function($x) use ($g){
                $y = $g->scale(3,$x);
                return $y;
            }
        );

        // build
        $x = $g->Variable($K->array([2,2]));
        $y = $nn->with($tape=$g->GradientTape(),$func,[$x],true);
        $grads = $tape->gradient($y,[$x]);
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->array([6,6]),
            $K->ndarray($y->value())));
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->array([3,3]),
            $K->ndarray($grads[0])));
        // exec
        $x = $g->Variable($K->array([2,2]));
        $y = $nn->with($tape=$g->GradientTape(),$func,[$x],true);
        $grads = $tape->gradient($y,[$x]);
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->array([6,6]),
            $K->ndarray($y->value())));
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->array([3,3]),
            $K->ndarray($grads[0])));
    }

    public function testNDArray()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $func = $g->Function(
            function($a,$x) use ($g){
                $y = $g->scale($a,$x);
                return $y;
            }
        );

        // build
        $a = $g->Variable($K->array(3));
        $x = $g->Variable($K->array([2,2]));
        $y = $nn->with($tape=$g->GradientTape(),$func,[$a,$x],true);
        $grads = $tape->gradient($y,[$a,$x]);
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->array([6,6]),
            $K->ndarray($y->value())));
        $this->assertEquals(4,$K->scalar($grads[0]));
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->array([3,3]),
            $K->ndarray($grads[1])));
        
        // exec
        $a = $g->Variable($K->array(3));
        $x = $g->Variable($K->array([2,2]));
        $y = $nn->with($tape=$g->GradientTape(),$func,[$a,$x],true);
        $grads = $tape->gradient($y,[$a,$x]);
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->array([6,6]),
            $K->ndarray($y->value())));
        $this->assertEquals(4,$K->scalar($grads[0]));
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->array([3,3]),
            $K->ndarray($grads[1])));
    }
}
