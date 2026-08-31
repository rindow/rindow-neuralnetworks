<?php
namespace RindowTest\NeuralNetworks\Gradient\Core\ArraySpecTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Gradient\ArrayShape;
use Rindow\NeuralNetworks\Model\AbstractModel;
use Interop\Polite\Math\Matrix\Buffer;
use Interop\Polite\Math\Matrix\NDArray;

class ArraySpecTest extends TestCase
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

    public function testShapeAndDtype()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $spec = $g->ArraySpec([1,2,3],dtype:NDArray::float32);
        $this->assertEquals([1,2,3],$spec->shape()->toArray());
        $this->assertEquals(NDArray::float32,$spec->dtype());
    }
}