<?php
namespace RindowTest\NeuralNetworks\Gradient\Func\Log1pTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

class Log1pTest extends TestCase
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

        $x = $g->Variable($K->array(1.5e-8));
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x){
                $y = $g->log1p($x);
                return $y;
            }
        );

        $this->assertTrue($mo->la()->isclose($mo->array(1.5e-8),$K->ndarray($y->value())));
        $this->assertTrue($mo->la()->isclose($mo->array(1.0),$K->ndarray($tape->gradient($y,$x))));
    }
}
