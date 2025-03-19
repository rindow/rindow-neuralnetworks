<?php
namespace RindowTest\NeuralNetworks\Gradient\Layer\GraphLayerTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

class GraphLayerTest extends TestCase
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

    public function testMultiUseLayer()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x = $g->Variable($K->array([
            [1,2],
            [3,4],
        ]),name:'x');
        $flatten = $nn->layers->Flatten(name:'flatten');
        $add = $nn->layers->Add(name:'add');

        $z = $nn->with($tape=$g->GradientTape(),
            function() use ($flatten,$add,$x) {
                $y0 = $flatten($x,true);
                $y0->setName('y0');
                $y1 = $flatten($x,true);
                $y1->setName('y1');
                $z = $add([$y0,$y1],true);
                $z->setName('z');
                return $z;
            }
        );
        $dx = $tape->gradient($z, $x);
        $this->assertEquals([2,2],$dx->shape());
        $this->assertEquals([[2,2],[2,2]],$dx->toArray());
    }
}
