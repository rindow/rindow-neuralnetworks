<?php
namespace RindowTest\NeuralNetworks\Gradient\Core\VariableTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Gradient\Core\Variable;
use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\Buffer;

class VariableTest extends TestCase
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

    public function testVariableNormal()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $a = $g->Variable($K->array([1,2]));
        $this->assertEquals("[1,2]",$K->toString($a->value()));
        $this->assertEquals(NDArray::float32,$a->dtype());
        $this->assertEquals(2,$a->size());
        $this->assertEquals([2],$a->shape());
        $this->assertEquals(1,$a->ndim());
        $this->assertEquals(0,$a->offset());
        if($mo->isAdvanced()) {
            $this->assertInstanceof(Buffer::class,$a->buffer());
        } else {
            $this->assertInstanceof(\SplFixedArray::class,$a->buffer());
        }
 
        $a2 = $a->reshape([2,1]);
        $this->assertInstanceof(Variable::class,$a2);
        $this->assertEquals("[[1],[2]]",$K->toString($a2->value()));
        $this->assertEquals([2,1],$a2->shape());
        $this->assertEquals(2,$a2->ndim());
        $this->assertEquals(NDArray::float32,$a2->dtype());
    }

    public function testApplyBackendFunction()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $a = $g->Variable(1);
        $b = $g->Variable(2);
        $c = $K->add($a,$b);
        $this->assertEquals(3,$K->scalar($c));
    }

    public function testArrayAccess()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $a = $g->Variable([1,2,3]);
        $this->assertEquals(1,$K->scalar($a[0]));
        $this->assertEquals(2,$K->scalar($a[1]));
        $this->assertEquals(3,$K->scalar($a[2]));
    }

    public function testIterableAccess()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $a = $g->Variable([1,2,3]);
        $this->assertCount(3,$a);
        $n = 0;
        $results = [];
        foreach($a as $i => $v) {
            $n++;
            $results[] = "a[$i]=".$K->scalar($v);
        }
        $this->assertEquals(3,$n);
        $this->assertEquals([
            "a[0]=1",
            "a[1]=2",
            "a[2]=3",
        ],$results);
    }
}