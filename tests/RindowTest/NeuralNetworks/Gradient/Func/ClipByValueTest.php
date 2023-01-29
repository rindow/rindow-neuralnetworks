<?php
namespace RindowTest\NeuralNetworks\Gradient\Func\ClipByValueTest;

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

    public function testSingleValue()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x = $g->Variable($K->array([0.2, 0.4, 0.6]));
        $c = $g->Variable($K->array(2));
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x,$c){
                $y = $g->clipByValue($x, 0.3, 0.5);
                $y = $g->mul($y,$c);
                return $y;
            }
        );
        $grad = $tape->gradient($y,$x);
        $this->assertTrue($mo->la()->isclose($mo->array([0.6, 0.8, 1.0]),$K->ndarray($y->value())));
        $this->assertTrue($mo->la()->isclose($mo->array([0,2,0]),$K->ndarray($grad)));
    }
}
