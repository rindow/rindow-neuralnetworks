<?php
namespace RindowTest\NeuralNetworks\Gradient\Func\LogTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

class LogTest extends TestCase
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

        $x = $g->Variable($K->array(exp(1.0)));
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x){
                $y = $g->log($x);
                return $y;
            }
        );

        $this->assertTrue($mo->la()->isclose($mo->array(1.0),$K->ndarray($y->value())));
        $this->assertTrue($mo->la()->isclose($mo->array(0.36787945),$K->ndarray($tape->gradient($y,$x))));
    }
}
