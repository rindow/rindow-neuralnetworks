<?php
namespace RindowTest\NeuralNetworks\Gradient\Func\MaskingTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;

class MaskingTest extends TestCase
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

        $mask = $g->constant([
            [true, true, false],
            [false,true, true],
        ],dtype:NDArray::bool);
        $x = $g->Variable($K->array([
            [10.0, 20.0, 30.0],
            [0.5,  1.5,  2.5 ]
        ],dtype:NDArray::float32));
        $salt = $mo->la()->range(start:1,limit:1+array_product($x->shape()),dtype:NDArray::float32)
                ->reshape($x->shape());
        $salt = $g->Variable($salt);

        [$outputs,$y] = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$mask,$x,$salt){
                $y = $g->masking($mask,$x,fill:999);
                $outputs = $g->mul($y,$salt);
                return [$outputs,$y];
            }
        );
        $dx = $tape->gradient($outputs,$x);

        $y = $K->ndarray($y->value());
        $truesY = $mo->array([
            [10.0, 20.0, 999],
            [999,  1.5,  2.5],
        ]);
        $this->assertTrue($mo->la()->isclose(
            $y,
            $truesY,
        ));

        $truesDx = $mo->array([
            [1, 2, 0],
            [0, 5, 6],
        ]);
        $this->assertTrue($mo->la()->isclose(
            $dx,
            $truesDx,
        ));
    }
}
