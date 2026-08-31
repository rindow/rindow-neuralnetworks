<?php
namespace RindowTest\NeuralNetworks\Gradient\Func\L2NormTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

class L2NormTest extends TestCase
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
        [$z, $y] = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x,$c){
                $y = $g->l2norm($x);
                $z = $g->mul($y,$c);
                return [$z, $y];
            }
        );
        //echo $K->toString($y)."\n";
        $grads = $tape->gradient($z,$x);
        //echo $K->toString($grads)."\n";

        $this->assertTrue($mo->la()->isclose($mo->array(5.47722578),$K->ndarray($y->value())));
        $this->assertTrue($mo->la()->isclose($mo->array([
            0.36514836,
            0.73029673,
            1.09544515,
            1.46059346
        ]),$K->ndarray($grads)));

    }

    public function testAxis0()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x = $g->Variable($K->array([[1,2],[3,4]]));
        $c = $g->Variable($K->array([2,4]));
        [$z,$y] = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x,$c){
                $y = $g->l2norm($x,axis:0);
                return [$g->mul($y,$c),$y];
            }
        );
        //echo $K->toString($y)."\n";
        $grads = $tape->gradient($z,$x);
        //echo $K->toString($grads)."\n";

        $this->assertTrue($mo->la()->isclose($mo->array([3.1622777, 4.472136]),$K->ndarray($y->value())));
        $this->assertTrue($mo->la()->isclose($mo->array([
            [0.6324555, 1.7888544],
            [1.8973665, 3.5777087],
        ]),$K->ndarray($grads)));

    }

    public function testAxisLast()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x = $g->Variable($K->array([[1,2],[3,4]]));
        $c = $g->Variable($K->array([2,4]));
        [$z,$y] = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x,$c){
                $y = $g->l2norm($x,axis:-1);
                return [$g->mul($y,$c),$y];
            }
        );
        //echo $K->toString($y)."\n";
        $grads = $tape->gradient($z,$x);
        //echo $K->toString($grads)."\n";

        $this->assertTrue($mo->la()->isclose($mo->array([2.236068, 5.0]),$K->ndarray($y->value())));
        $this->assertTrue($mo->la()->isclose($mo->array([
            [0.8944272, 1.7888544],
            [2.4      , 3.2      ],
        ]),$K->ndarray($grads)));

    }
}
