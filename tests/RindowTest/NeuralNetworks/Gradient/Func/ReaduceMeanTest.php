<?php
namespace RindowTest\NeuralNetworks\Gradient\Func\ReduceMeanTest;

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

    public function testAxisNull()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x = $g->Variable($K->array([1,2,3,4]));
        $c = $g->Variable($K->array(2));
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x,$c){
                $y = $g->reduceMean($x);
                return $g->mul($y,$c);
            }
        );
        //echo $K->toString($y)."\n";
        $grads = $tape->gradient($y,$x);
        //echo $K->toString($grads)."\n";

        $this->assertTrue($mo->la()->isclose($mo->array(5),$K->ndarray($y->value())));
        $this->assertTrue($mo->la()->isclose($mo->array([0.5,0.5,0.5,0.5]),$K->ndarray($grads)));

    }

    public function testAxis0()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x = $g->Variable($K->array([[1,2],[3,4]]));
        $c = $g->Variable($K->array([2,4]));
        [$y,$y0] = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x,$c){
                $y = $g->reduceMean($x,axis:0);
                return [$g->mul($y,$c),$y];
            }
        );
        $grads = $tape->gradient($y,$x);

        $this->assertTrue($mo->la()->isclose($mo->array([4,12]),$K->ndarray($y->value())));
        $this->assertTrue($mo->la()->isclose($mo->array([[1,2],[1,2]]),$K->ndarray($grads)));

    }
}
