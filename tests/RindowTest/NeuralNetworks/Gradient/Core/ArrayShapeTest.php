<?php
namespace RindowTest\NeuralNetworks\Gradient\Core\ArrayShapeTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Gradient\Core\Variable;
use Rindow\NeuralNetworks\Gradient\Core\ArrayShape;
use Interop\Polite\Math\Matrix\Buffer;

class ArrayShapeTest extends TestCase
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

    
    public function testCount()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $shape = new ArrayShape([1,2,3]);
        $a = $g->Variable($shape);
        $this->assertCount(3,$a);
        $this->assertCount(3,$a->value());
    }

    public function testArrayAccess()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $shape = new ArrayShape([1,2,3]);
        $a = $g->Variable($shape);
        $this->assertEquals(1,$a[0]);
        $this->assertEquals(2,$a[1]);
        $this->assertEquals(3,$a[2]);

        $this->assertEquals(1,$a->value()[0]);
        $this->assertEquals(2,$a->value()[1]);
        $this->assertEquals(3,$a->value()[2]);
    }

    public function testIterableAccess()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $shape = new ArrayShape([1,2,3]);
        $a = $g->Variable($shape);
        $this->assertCount(3,$a);
        $n = 0;
        $results = [];
        foreach($a as $i => $v) {
            $n++;
            $results[] = "a[$i]=".$v;
        }
        $this->assertEquals(3,$n);
        $this->assertEquals([
            "a[0]=1",
            "a[1]=2",
            "a[2]=3",
        ],$results);
    }
}