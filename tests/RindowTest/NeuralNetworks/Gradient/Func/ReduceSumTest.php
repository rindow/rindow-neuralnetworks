<?php
namespace RindowTest\NeuralNetworks\Gradient\Func\ReduceSumTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

class ReduceSumTest extends TestCase
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

    public function testAxisNull()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $func = $g->Function(
            function($x,$c) use ($g){
                $y = $g->reduceSum($x);
                return $g->mul($y,$c);
            }
        );

        // build
        $x = $g->Variable($K->array([1,2,3,4]));
        $c = $g->Variable($K->array(2));
        $y = $nn->with($tape=$g->GradientTape(),$func,[$x,$c],true);
        $grads = $tape->gradient($y,$x);
        $this->assertTrue($mo->la()->isclose($mo->array(20),$K->ndarray($y->value())));
        $this->assertTrue($mo->la()->isclose($mo->array([2,2,2,2]),$K->ndarray($grads)));
        // exec
        $x = $g->Variable($K->array([1,2,3,4]));
        $c = $g->Variable($K->array(2));
        $y = $nn->with($tape=$g->GradientTape(),$func,[$x,$c],true);
        $grads = $tape->gradient($y,$x);
        $this->assertTrue($mo->la()->isclose($mo->array(20),$K->ndarray($y->value())));
        $this->assertTrue($mo->la()->isclose($mo->array([2,2,2,2]),$K->ndarray($grads)));

    }

    public function testAxis0()
    {
        //echo "\n";
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        //echo "=====immedate test====\n";

        $x = $g->Variable($K->array([[1,2],[3,4]]));
        $c = $g->Variable($K->array([2,4]));
        [$y,$y0] = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x,$c){
                $y = $g->reduceSum($x,axis:0);
                return [$g->mul($y,$c),$y];
            }
        );
        //echo $K->toString($y)."\n";
        $grads = $tape->gradient($y,$x);
        //echo $K->toString($grads)."\n";

        $this->assertTrue($mo->la()->isclose($mo->array([8,24]),$K->ndarray($y->value())));
        $this->assertTrue($mo->la()->isclose($mo->array([[2,4],[2,4]]),$K->ndarray($grads)));

        //echo "=====function test====\n";


        //////////////////////

        $func = $g->Function(
            function($x,$c) use ($g){
                $y = $g->reduceSum($x,axis:0);
                return [$g->mul($y,$c),$y];
            }
        );

        $x = $g->Variable($K->array([[1,2],[3,4]]));
        $c = $g->Variable($K->array([2,4]));
        [$y,$y0] = $nn->with($tape=$g->GradientTape(),$func,[$x,$c],true);
        $grads = $tape->gradient($y,$x);
        //echo $mo->toString($grads,null,true);

        $this->assertTrue($mo->la()->isclose($mo->array([8,24]),$K->ndarray($y->value())));
        $this->assertTrue($mo->la()->isclose($mo->array([[2,4],[2,4]]),$K->ndarray($grads)));

    }
}
