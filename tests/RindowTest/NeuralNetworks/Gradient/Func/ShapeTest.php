<?php
namespace RindowTest\NeuralNetworks\Gradient\Func\ShapeTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Gradient\ArrayShape;
use Rindow\NeuralNetworks\Gradient\Variable;

class ShapeTest extends TestCase
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

    public function testValue()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        //// scalar value
        $func = $g->Function(
            function($x) use ($g){
                $y = $g->shape($x);
                return $y;
            }
        );
        $x = $g->Variable($K->array(9.0));
        // build
        $y = $nn->with($tape=$g->GradientTape(),$func,[$x],true);
        // exec
        $y = $nn->with($tape=$g->GradientTape(),$func,[$x],true);

        $this->assertInstanceof(Variable::class,$y);
        $this->assertInstanceof(ArrayShape::class,$y->value());
        $this->assertCount(0,$y->value());


        //// 2D array
        $func = $g->Function(
            function($x) use ($g){
                $y = $g->shape($x);
                return $y;
            }
        );
        $x = $g->Variable($K->ones([2,3]));
        // build
        $y = $nn->with($tape=$g->GradientTape(),$func,[$x],true);

        $this->assertInstanceof(Variable::class,$y);
        $this->assertInstanceof(ArrayShape::class,$y->value());
        $this->assertCount(2,$y);
        $this->assertEquals(2,$y[0]);
        $this->assertEquals(3,$y[1]);

        // exec
        $y = $nn->with($tape=$g->GradientTape(),$func,[$x],true);

        $this->assertInstanceof(Variable::class,$y);
        $this->assertInstanceof(ArrayShape::class,$y->value());
        $this->assertCount(2,$y);
        $this->assertEquals(2,$y[0]);
        $this->assertEquals(3,$y[1]);

    }
}
