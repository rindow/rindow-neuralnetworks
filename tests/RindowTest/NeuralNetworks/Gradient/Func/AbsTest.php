<?php
namespace RindowTest\NeuralNetworks\Gradient\Func\AbsTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

class AbsTest extends TestCase
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

        $x = $g->Variable($K->array([
            [-1.0, -0.5, 0.0, 0.5, 1.0],
            [-2.0, -1.0, 0.0, 1.0, 2.0],
        ]));
        $c = $g->Variable($K->array(2));
        [$y,$z] = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x,$c){
                $y = $g->abs($x);
                $z = $g->mul($y,$c);
                return [$y,$z];
            }
        );

        $this->assertTrue($mo->la()->isclose($mo->array([
            [ 1.0, 0.5, 0.0, 0.5, 1.0 ],
            [ 2.0, 1.0, 0.0, 1.0, 2.0 ]
        ]),$K->ndarray($y->value())));

        $grads = $tape->gradient($z,$x);
        $this->assertTrue($mo->la()->isclose($mo->array([
            [-2.0, -2.0, 0.0, 2.0, 2.0],
            [-2.0, -2.0, 0.0, 2.0, 2.0],
        ]),$K->ndarray($grads)));
    }
}
