<?php
namespace RindowTest\NeuralNetworks\Gradient\Core\ScalarTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Gradient\Core\Variable;
use Rindow\NeuralNetworks\Gradient\Core\Scalar;
use Interop\Polite\Math\Matrix\Buffer;

class ScalarTest extends TestCase
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

    public function testCreateFromInteger()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $scalar = new Scalar(10);
        $a = $g->Variable($scalar);
        $this->assertInstanceof(Scalar::class,$a->value());
        $this->assertEquals(10,$a->value()->value());
    }

    public function testCreateFromNDArray()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $scalar = new Scalar($mo->array(10));
        $a = $g->Variable($scalar);
        $this->assertInstanceof(Scalar::class,$a->value());
        $this->assertEquals(10,$a->value()->value());
    }
}
