<?php
namespace RindowTest\NeuralNetworks\Gradient\Func\IncrementTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

class IncrementTest extends TestCase
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

    public function testScalarNormal()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $func = $g->Function(
            function($x) use ($g){
                $y = $g->increment($x,3);
                return $y;
            }
        );

        // build
        $x = $g->Variable($K->array([2,2]));
        $y = $nn->with($tape=$g->GradientTape(),$func,[$x],true);
        $grads = $tape->gradient($y,[$x]);
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->array([5,5]),
            $K->ndarray($y->value())));
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->array([1,1]),
            $K->ndarray($grads[0])));
        // exec
        $x = $g->Variable($K->array([2,2]));
        $y = $nn->with($tape=$g->GradientTape(),$func,[$x],true);
        $grads = $tape->gradient($y,[$x]);
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->array([5,5]),
            $K->ndarray($y->value())));
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->array([1,1]),
            $K->ndarray($grads[0])));
    }

    public function testScalarWithAlpha()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $func = $g->Function(
            function($x) use ($g){
                $y = $g->increment($x,3,5);
                return $y;
            }
        );

        // build
        $x = $g->Variable($K->array([2,2]));
        $y = $nn->with($tape=$g->GradientTape(),$func,[$x],true);
        $grads = $tape->gradient($y,[$x]);
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->array([13,13]),
            $K->ndarray($y->value())));
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->array([5,5]),
            $K->ndarray($grads[0])));
        // exec
        $x = $g->Variable($K->array([2,2]));
        $y = $nn->with($tape=$g->GradientTape(),$func,[$x],true);
        $grads = $tape->gradient($y,[$x]);
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->array([13,13]),
            $K->ndarray($y->value())));
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->array([5,5]),
            $K->ndarray($grads[0])));
    }

    public function testNDArray()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $func = $g->Function(
            function($x,$b,$a) use ($g){
                $y = $g->increment($x,$b,$a);
                return $y;
            }
        );

        // build
        $x = $g->Variable($K->array([2,2]));
        $b = $g->Variable($K->array(3));
        $a = $g->Variable($K->array(5));
        $y = $nn->with($tape=$g->GradientTape(),$func,[$x,$b,$a],true);
        $grads = $tape->gradient($y,[$x,$b,$a]);
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->array([13,13]),
            $K->ndarray($y->value())));
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->array([5,5]),
            $K->ndarray($grads[0])));
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->array(2),
            $K->ndarray($grads[1])));
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->array(4),
            $K->ndarray($grads[2])));
        
        // exec
        $x = $g->Variable($K->array([2,2]));
        $b = $g->Variable($K->array(3));
        $a = $g->Variable($K->array(5));
        $y = $nn->with($tape=$g->GradientTape(),$func,[$x,$b,$a],true);
        $grads = $tape->gradient($y,[$x,$b,$a]);
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->array([13,13]),
            $K->ndarray($y->value())));
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->array([5,5]),
            $K->ndarray($grads[0])));
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->array(2),
            $K->ndarray($grads[1])));
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->array(4),
            $K->ndarray($grads[2])));
    }
}
