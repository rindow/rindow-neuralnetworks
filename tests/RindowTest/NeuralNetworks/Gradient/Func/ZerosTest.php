<?php
namespace RindowTest\NeuralNetworks\Gradient\Func\ZerosTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Gradient\Core\Scalar;

class ZerosTest extends TestCase
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

    public function testArrayShape()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        //// 
        $func = $g->Function(
            function($x) use ($g){
                $y0 = $g->zeros($g->shape($x),dtype:NDArray::float32);
                $y1 = $g->zeros($g->shape($x),dtype:NDArray::int32);
                return [$y0,$y1];
            }
        );
        // build
        $x = $g->Variable($K->zeros([1,2,3,4],NDArray::float32));
        [$y0,$y1] = $nn->with($tape=$g->GradientTape(),$func,[$x],true);

        $this->assertInstanceof(Variable::class,$y0);
        $this->assertEquals(NDArray::float32,$y0->dtype());
        $this->assertInstanceof(NDArray::class,$y0->value());
        $this->assertEquals([1,2,3,4],$y0->value()->shape());
        $this->assertInstanceof(Variable::class,$y1);
        $this->assertEquals(NDArray::int32,$y1->dtype());
        $this->assertInstanceof(NDArray::class,$y1->value());
        $this->assertEquals([1,2,3,4],$y1->value()->shape());

        // exec
        $x = $g->Variable($K->zeros([4,3,2,1],NDArray::float32));
        [$y0,$y1] = $nn->with($tape=$g->GradientTape(),$func,[$x],true);

        $this->assertInstanceof(Variable::class,$y0);
        $this->assertEquals(NDArray::float32,$y0->dtype());
        $this->assertInstanceof(NDArray::class,$y0->value());
        $this->assertEquals([4,3,2,1],$y0->value()->shape());
        $this->assertInstanceof(Variable::class,$y1);
        $this->assertEquals(NDArray::int32,$y1->dtype());
        $this->assertInstanceof(NDArray::class,$y1->value());
        $this->assertEquals([4,3,2,1],$y1->value()->shape());

        $y = $K->ndarray($y1);
        $this->assertEquals(0,$y[0][0][0][0]);
    }

    public function testPhpArray()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        //// 
        $func = $g->Function(
            function($a,$b,$c) use ($g){
                $y0 = $g->zeros([1,$a,$b,$c],dtype:NDArray::float32);
                $y1 = $g->zeros([1,$c,$b,$a],dtype:NDArray::int32);
                return [$y0,$y1];
            }
        );
        // build
        $a = $g->Variable(new Scalar(2));
        $b = $g->Variable(new Scalar(3));
        $c = $g->Variable(new Scalar(4));
        [$y0,$y1] = $nn->with($tape=$g->GradientTape(),$func,[$a,$b,$c],true);

        $this->assertInstanceof(Variable::class,$y0);
        $this->assertEquals(NDArray::float32,$y0->dtype());
        $this->assertInstanceof(NDArray::class,$y0->value());
        $this->assertEquals([1,2,3,4],$y0->value()->shape());
        $this->assertInstanceof(Variable::class,$y1);
        $this->assertEquals(NDArray::int32,$y1->dtype());
        $this->assertInstanceof(NDArray::class,$y1->value());
        $this->assertEquals([1,4,3,2],$y1->value()->shape());

        // exec
        $a = $g->Variable(new Scalar(4));
        $b = $g->Variable(new Scalar(3));
        $c = $g->Variable(new Scalar(2));
        [$y0,$y1] = $nn->with($tape=$g->GradientTape(),$func,[$a,$b,$c],true);

        $this->assertInstanceof(Variable::class,$y0);
        $this->assertEquals(NDArray::float32,$y0->dtype());
        $this->assertInstanceof(NDArray::class,$y0->value());
        $this->assertEquals([1,4,3,2],$y0->value()->shape());
        $this->assertInstanceof(Variable::class,$y1);
        $this->assertEquals(NDArray::int32,$y1->dtype());
        $this->assertInstanceof(NDArray::class,$y1->value());
        $this->assertEquals([1,2,3,4],$y1->value()->shape());

        $y = $K->ndarray($y1);
        $this->assertEquals(0,$y[0][0][0][0]);
    }
}
