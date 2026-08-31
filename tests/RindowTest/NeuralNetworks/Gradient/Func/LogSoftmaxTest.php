<?php
namespace RindowTest\NeuralNetworks\Gradient\Func\LogSoftmaxTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;

class LogSoftmaxTest extends TestCase
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

    public function testNormal()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x = $g->Variable($K->array([
            [ 1.0,  2.0,  3.0,  4.0],
            [10.0,  5.0,  1.0, -5.0],
        ],dtype:NDArray::float32));

        $c = $g->constant([
            [ 1.7640524,   0.4001572,   0.978738  ,  2.2408931 ],
            [ 1.867558 ,  -0.9772779,   0.95008844, -0.1513572 ],
        ]);
        [$y,$z] = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x,$c){
                $y = $g->logSoftmax($x); // always axis=-1
                $z = $g->mul($y,$c);
                return [$y,$z];
            }
        );
        $dx = $tape->gradient($z,$x);

        $y = $K->ndarray($y->value());
        $truesY = $mo->array([
            [-3.4401898e+00, -2.4401898e+00, -1.4401897e+00, -4.4018975e-01],
            [-6.8382523e-03, -5.0068383e+00, -9.0068378e+00, -1.5006838e+01],
        ]);
        $this->assertTrue($mo->la()->isclose(
            $y,
            $truesY,
        ));

        $truesDx = $mo->array([
            [ 1.591454  , -0.06901383, -0.29660124, -1.225838  ],
            [ 0.19005644, -0.98858076,  0.94988143, -0.15135771],
        ]);
        $this->assertTrue($mo->la()->isclose(
            $dx,
            $truesDx,
        ));
    }
}
