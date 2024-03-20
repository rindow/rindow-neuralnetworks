<?php
namespace RindowTest\NeuralNetworks\Gradient\Func\CastTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;

class CastTest extends TestCase
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

    public function testIntToFloat()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $func = $g->Function(
            function($x) use ($g){
                $y = $g->cast($x,NDArray::float32);
                return $y;
            }
        );

        // build
        $x = $g->Variable($K->array([3,0],NDArray::int32));
        $y = $nn->with($tape=$g->GradientTape(),$func,[$x],true);
        $this->assertEquals(NDArray::float32,$y->dtype());
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->array([3,0]),
            $K->ndarray($y->value())));
        // exec
        $x = $g->Variable($K->array([3,0],NDArray::int32));
        $y = $nn->with($tape=$g->GradientTape(),$func,[$x],true);
        $this->assertEquals(NDArray::float32,$y->dtype());
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->array([3,0]),
            $K->ndarray($y->value())));
    }

    public function testIntToBoolToInt()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $func = $g->Function(
            function($x) use ($g){
                $y = $g->cast($x,NDArray::bool);
                $z = $g->cast($y,NDArray::int32);
                return [$y,$z];
            }
        );

        // build
        $x = $g->Variable($K->array([3,0],NDArray::int32));
        [$y,$z] = $nn->with($tape=$g->GradientTape(),$func,[$x],true);
        $this->assertEquals(NDArray::bool,$y->dtype());
        $this->assertEquals([true,false],$y->toArray());
        $this->assertEquals(NDArray::int32,$z->dtype());
        $this->assertEquals([1,0],$z->toArray());
        // exec
        $x = $g->Variable($K->array([3,0],NDArray::int32));
        [$y,$z] = $nn->with($tape=$g->GradientTape(),$func,[$x],true);
        $this->assertEquals(NDArray::bool,$y->dtype());
        $this->assertEquals([true,false],$y->toArray());
        $this->assertEquals(NDArray::int32,$z->dtype());
        $this->assertEquals([1,0],$z->toArray());
    }

}
