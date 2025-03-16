<?php
namespace RindowTest\NeuralNetworks\Gradient\Core\MaskedNDArrayTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Matrix\Drivers\Service;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Gradient\Core\MaskedNDArray;
use Rindow\NeuralNetworks\Gradient\MaskedNDArray as MaskedNDArrayInterface;
use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\Buffer;

class MaskedNDArrayTest extends TestCase
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

    protected function maskedValue(NDArray $value, NDArray $mask) : MaskedNDArrayInterface
    {
        return new MaskedNDArray($value,$mask);
    }

    public function testVariableNormal()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);

        $value = $K->array([1,2,3,4], NDArray::float32);
        $mask = $K->array([true,false,true,false], NDArray::bool);
        $a = $this->maskedValue($value,$mask);
        $b = $K->array([2,3,4,5], NDArray::float32);

        $c = $K->add($a,$b);
        $this->assertEquals([3,5,7,9], $c->toArray());
        $this->assertNotEquals(MaskedNDArray::class, get_class($c));

        $this->assertEquals(spl_object_id($value),spl_object_id($a->value()));
        $this->assertEquals(spl_object_id($mask), spl_object_id($a->mask()));
    }

    public function testNDArrayMethods()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);

        $value = $K->array([[1,2],[3,4]], NDArray::float32);
        $mask = $K->array([[true,false],[true,false]], NDArray::bool);
        $a = $this->maskedValue($value,$mask);

        $this->assertEquals([2,2], $a->shape());
        $this->assertEquals(2,   $a->ndim());
        $this->assertEquals(NDArray::float32, $a->dtype());
        $this->assertInstanceOf(Buffer::class, $a->buffer());
        $this->assertEquals(0, $a->offset());
        $this->assertEquals(4, $a->size());
        $this->assertEquals([1,4], $a->reshape([1,4])->shape());
        $this->assertNotEquals(MaskedNDArray::class, get_class($a->reshape([1,4])));
        $this->assertEquals([[1,2],[3,4]], $a->toArray());
    }

    public function testExtraNDArrayMethods()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);

        $value = $K->array([[1,2],[3,4]], NDArray::float32);
        $mask = $K->array([[true,false],[true,false]], NDArray::bool);
        $a = $this->maskedValue($value,$mask);

        $this->assertEquals(2, $a->count());
        $this->assertInstanceOf(Service::class, $a->service());
    }

    public function testNDArrayClone()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);

        $value = $K->array([1,2,3,4], NDArray::float32);
        $mask = $K->array([true,false,true,false], NDArray::bool);
        $a = $this->maskedValue($value,$mask);

        $c = clone $a;
        $this->assertEquals(MaskedNDArray::class, get_class($c));
    }

}
