<?php
namespace RindowTest\NeuralNetworks\Gradient\Func\GreaterTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

class GreaterTest extends TestCase
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
                $y = $g->greater($x,2);
                return $y;
            }
        );

        // build
        $x = $g->Variable($K->array([3,1]));
        $y = $nn->with($tape=$g->GradientTape(),$func,[$x],true);
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->array([1,0]),
            $K->ndarray($y->value())));
        // exec
        $x = $g->Variable($K->array([3,1]));
        $y = $nn->with($tape=$g->GradientTape(),$func,[$x],true);
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->array([1,0]),
            $K->ndarray($y->value())));
    }

    public function testNDArray()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $func = $g->Function(
            function($a,$x) use ($g){
                $y = $g->greater($x,$a);
                return $y;
            }
        );

        // build
        $a = $g->Variable($K->array(2));
        $x = $g->Variable($K->array([3,1]));
        $y = $nn->with($tape=$g->GradientTape(),$func,[$a,$x],true);
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->array([1,0]),
            $K->ndarray($y->value())));
        // exec
        $a = $g->Variable($K->array(2));
        $x = $g->Variable($K->array([3,1]));
        $y = $nn->with($tape=$g->GradientTape(),$func,[$a,$x],true);
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->array([1,0]),
            $K->ndarray($y->value())));
    }
}
