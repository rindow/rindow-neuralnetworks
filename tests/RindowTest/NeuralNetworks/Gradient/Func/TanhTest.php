<?php
namespace RindowTest\NeuralNetworks\Gradient\Func\TanhTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

class TanhTest extends TestCase
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
            [-1.0,-0.5,0.0,0.5,1.0],
            [-2.0,-1.0,0.0,1.0,2.0],
        ]));
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x){
                $y = $g->tanh($x);
                return $y;
            }
        );

        $this->assertTrue($mo->la()->isclose($mo->array([
            [-0.7615942 , -0.46211717,  0.        ,  0.46211717,  0.7615942 ],
            [-0.9640276 , -0.7615942 ,  0.        ,  0.7615942 ,  0.9640276 ]
        ]),$K->ndarray($y->value())));
        $this->assertTrue($mo->la()->isclose($mo->array([
            [0.41997433, 0.7864477 , 1.        , 0.7864477 , 0.41997433],
            [0.07065082, 0.41997433, 1.        , 0.41997433, 0.07065082],
        ]),$K->ndarray($tape->gradient($y,$x))));
    }
}
