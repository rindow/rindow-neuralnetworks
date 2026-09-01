<?php
namespace RindowTest\NeuralNetworks\Gradient\Func\WhereTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;

class WhereTest extends TestCase
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

    public function testSameShape()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $condition = $g->Variable($K->array([
            [true, false, true],
            [false, true, false]
        ],dtype:NDArray::bool));
        $x = $g->Variable($K->array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ],dtype:NDArray::float32));
        $y = $g->Variable($K->array([
            [11.0, 12.0, 13.0],
            [14.0, 15.0, 16.0],
        ]));
        
        $salt = $g->Variable(2);

        [$z,$outputs] = $nn->with($tape=$g->GradientTape(),
            function() use ($condition,$x,$y,$salt,$g) {
                $outputs = $g->where($condition,$x,$y);
                $z = $g->mul($outputs,$salt);
                return [$z,$outputs];
            }
        );
        $grads = $tape->gradient($z,[$x,$y]);

        $outputs = $K->ndarray($outputs->value());
        $truesOutputs = $mo->array([
            [1.0, 12.0, 3.0],
            [14.0, 5.0, 16.0],
        ]);
        $this->assertTrue($mo->la()->isclose(
            $outputs,
            $truesOutputs,
        ));

        $truesDx = $mo->array([
            [2.0, 0.0, 2.0],
            [0.0, 2.0, 0.0],
        ]);
        $this->assertTrue($mo->la()->isclose(
            $grads[0],
            $truesDx,
        ));

        $truesDy = $mo->array([
            [0.0, 2.0, 0.0],
            [2.0, 0.0, 2.0],
        ]);
        $this->assertTrue($mo->la()->isclose(
            $grads[1],
            $truesDy,
        ));
    }

}
