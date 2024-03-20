<?php
namespace RindowTest\NeuralNetworks\Gradient\Func\SqrtTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

class SqrtTest extends TestCase
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

        $x = $g->Variable($K->array(9.0));
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x){
                $y = $g->sqrt($x);
                return $y;
            }
        );

        $this->assertTrue($mo->la()->isclose($mo->array(3.0),$K->ndarray($y->value())));
        $this->assertTrue($mo->la()->isclose($mo->array(1.0/(2.0*sqrt(9.0))),$K->ndarray($tape->gradient($y,$x))));

    }
}
