<?php
namespace RindowTest\NeuralNetworks\Gradient\Func\GetTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Gradient\ArrayShape;
use Rindow\NeuralNetworks\Gradient\Scalar;
use Rindow\NeuralNetworks\Gradient\Variable;

class GetTest extends TestCase
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

    public function testNDArraySingle()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        //// 1D array
        $func = $g->Function(
            function($x) use ($g){
                $y = $g->get($x,1);
                return $y;
            }
        );
        $x = $g->Variable($K->array([1,2,3,4]));
        // build
        $y = $nn->with($tape=$g->GradientTape(),$func,[$x],true);

        $this->assertInstanceof(Variable::class,$y);
        $this->assertInstanceof(Scalar::class,$y->value());
        $this->assertEquals(2,$y->value()->value());

        // exec
        $y = $nn->with($tape=$g->GradientTape(),$func,[$x],true);

        $this->assertInstanceof(Variable::class,$y);
        $this->assertInstanceof(Scalar::class,$y->value());
        $this->assertEquals(2,$y->value()->value());


        //// 2D array
        $func = $g->Function(
            function($x) use ($g){
                $y = $g->get($x,1);
                return $y;
            }
        );
        $x = $mo->arange(6)->reshape([2,3]);
        $x = $g->Variable($K->array($x));
        // build
        $y = $nn->with($tape=$g->GradientTape(),$func,[$x],true);
        // exec
        $y = $nn->with($tape=$g->GradientTape(),$func,[$x],true);

        $this->assertInstanceof(Variable::class,$y);
        $this->assertInstanceof(NDArray::class,$y->value());
        $this->assertEquals(1,$y->ndim());
        $this->assertEquals([3],$y->shape());
        $this->assertEquals([3,4,5],$y->value()->toArray());
    }

    public function testNDArrayRange()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        //// 1D array
        $func = $g->Function(
            function($x) use ($g){
                $y = $g->get($x,1,2);
                return $y;
            }
        );
        $x = $g->Variable($K->array([1,2,3,4]));
        // build
        $y = $nn->with($tape=$g->GradientTape(),$func,[$x],true);

        $this->assertInstanceof(Variable::class,$y);
        $this->assertInstanceof(NDArray::class,$y->value());
        $this->assertEquals(1,$y->ndim());
        $this->assertEquals([2],$y->shape());
        $this->assertEquals([2,3],$y->value()->toArray());

        // exec
        $y = $nn->with($tape=$g->GradientTape(),$func,[$x],true);

        $this->assertInstanceof(Variable::class,$y);
        $this->assertInstanceof(NDArray::class,$y->value());
        $this->assertEquals(1,$y->ndim());
        $this->assertEquals([2],$y->shape());
        $this->assertEquals([2,3],$y->value()->toArray());


        //// 2D array
        $func = $g->Function(
            function($x) use ($g){
                $y = $g->get($x,1,2);
                return $y;
            }
        );
        $x = $mo->arange(6)->reshape([3,2]);
        $x = $g->Variable($K->array($x));
        // build
        $y = $nn->with($tape=$g->GradientTape(),$func,[$x],true);
        // exec
        $y = $nn->with($tape=$g->GradientTape(),$func,[$x],true);

        $this->assertInstanceof(Variable::class,$y);
        $this->assertInstanceof(NDArray::class,$y->value());
        $this->assertEquals(2,$y->ndim());
        $this->assertEquals([2,2],$y->shape());
        $this->assertEquals([[2,3],[4,5]],$y->value()->toArray());
    }

    public function testArrayShapeSingle()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        //// 2D array
        $func = $g->Function(
            function($x) use ($g){
                $shape = $g->shape($x);
                $y = $g->get($shape,1);
                return $y;
            }
        );
        $x = $g->Variable($K->zeros([2,3]));
        // build
        $y = $nn->with($tape=$g->GradientTape(),$func,[$x],true);
        // exec
        $y = $nn->with($tape=$g->GradientTape(),$func,[$x],true);

        $this->assertInstanceof(Variable::class,$y);
        $this->assertInstanceof(Scalar::class,$y->value());
        $this->assertEquals(3,$y->value()->value());
    }

    public function testArrayShapeRange()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        //// 2D array
        $func = $g->Function(
            function($x) use ($g){
                $shape = $g->shape($x);
                $y = $g->get($shape,1,2);
                return $y;
            }
        );
        $x = $g->Variable($K->zeros([2,3,4]));
        // build
        $y = $nn->with($tape=$g->GradientTape(),$func,[$x],true);
        // exec
        $y = $nn->with($tape=$g->GradientTape(),$func,[$x],true);

        $this->assertInstanceof(Variable::class,$y);
        $this->assertInstanceof(ArrayShape::class,$y->value());
        $this->assertCount(2,$y->value());
        $this->assertEquals(3,$y->value()[0]);
        $this->assertEquals(4,$y->value()[1]);
    }
}
