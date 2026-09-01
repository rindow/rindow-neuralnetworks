<?php
namespace RindowTest\NeuralNetworks\Gradient\Func\SigmoidTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

class SigmoidTest extends TestCase
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
                $y = $g->sigmoid($x);
                $z = $g->mul($y,$c);
                return [$y,$z];
            }
        );

        $this->assertTrue($mo->la()->isclose($mo->array([
            [ 0.26894143, 0.37754067, 0.5       , 0.62245933, 0.73105858],
            [ 0.11920292, 0.26894143, 0.5       , 0.73105858, 0.88079708]
        ]),$K->ndarray($y->value())));

        $grads = $tape->gradient($z,$x);
        $this->assertTrue($mo->la()->isclose($mo->array([
            [0.39322385, 0.47000742,    0.5       ,  0.47000742,  0.3932239 ],
            [0.2099872 , 0.39322385,    0.5       ,  0.39322385,  0.20998715],
        ]),$K->ndarray($grads)));
    }
}
