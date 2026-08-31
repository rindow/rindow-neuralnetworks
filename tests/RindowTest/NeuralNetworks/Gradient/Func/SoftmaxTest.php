<?php
namespace RindowTest\NeuralNetworks\Gradient\Func\SoftmaxTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;

class SoftmaxTest extends TestCase
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

    public function testLossOne()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x = $g->Variable($K->array([
            [1.0, 2.0, 3.0],
            [0.5, 1.5, 2.5]
        ],dtype:NDArray::float32));

        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x){
                $y = $g->softmax($x); // always axis=-1
                return $y;
            }
        );
        $dx = $tape->gradient($y,$x);

        $y = $K->ndarray($y->value());
        $truesY = $mo->array([
            [0.090031, 0.244728, 0.665241],
            [0.090031, 0.244728, 0.665241],
        ]);
        $this->assertTrue($mo->la()->isclose(
            $y,
            $truesY,
        ));

        $truesDx = $mo->array([
            [0, 0, 0],
            [0, 0, 0],
        ]);
        $this->assertTrue($mo->la()->isclose(
            $dx,
            $truesDx,
        ));
    }

    public function testNormal()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x = $g->Variable($K->array([
            [1.0, 2.0, 3.0],
            [0.5, 1.5, 2.5]
        ],dtype:NDArray::float32));
        $salt = $mo->la()->range(start:1,limit:1+array_product($x->shape()),dtype:NDArray::float32)
                ->reshape($x->shape());
        $salt = $g->Variable($salt);

        [$outputs,$y] = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x,$salt){
                $y = $g->softmax($x); // always axis=-1
                $outputs = $g->mul($y,$salt);
                return [$outputs,$y];
            }
        );
        $dx = $tape->gradient($outputs,$x);

        $y = $K->ndarray($y->value());
        $truesY = $mo->array([
            [0.090031, 0.244728, 0.665241],
            [0.090031, 0.244728, 0.665241],
        ]);
        $this->assertTrue($mo->la()->isclose(
            $y,
            $truesY,
        ));

        $truesDx = $mo->array([
            [-0.1418171 , -0.14077035,  0.28258747],
            [-0.14181706, -0.14077029,  0.28258765],
        ]);
        $this->assertTrue($mo->la()->isclose(
            $dx,
            $truesDx,
        ));
    }
}
