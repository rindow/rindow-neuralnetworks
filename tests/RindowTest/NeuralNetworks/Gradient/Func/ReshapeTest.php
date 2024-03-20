<?php
namespace RindowTest\NeuralNetworks\Gradient\Func\ReshapeTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

class ReshapeTest extends TestCase
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

    public function testSingleValueNormal()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        /// reshape [] => [1]
        $x = $g->Variable($K->array(10));
        $z = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x){
                $z = $g->reshape($x,[1]);
                return $z;
            }
        );
        $this->assertEquals("[10]",$mo->toString($K->ndarray($z->value())));
        $this->assertTrue($x->isbackpropagatable());
        $this->assertTrue($z->isbackpropagatable());
        $dx = $tape->gradient($z,$x);
        $this->assertEquals("1",$mo->toString($K->ndarray($dx)));

        /// reshape [1] => []
        $x = $g->Variable($K->array([10]));
        $z = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x){
                $z = $g->reshape($x,[]);
                return $z;
            }
        );
        $this->assertEquals("10",$mo->toString($K->ndarray($z->value())));
        $this->assertTrue($x->isbackpropagatable());
        $this->assertTrue($z->isbackpropagatable());
        $dx = $tape->gradient($z,$x);
        $this->assertEquals("[1]",$mo->toString($K->ndarray($dx)));
    }

    public function testSingleValueIllegalShape()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $this->expectException(InvalidArgumentException::class);

        /// reshape [2] => []
        $x = $g->Variable($K->array([10,20]));
        $z = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x){
                $z = $g->reshape($x,[1]);
                return $z;
            }
        );
    }

    public function testNormal()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        // reshape [3,2] => [1,2,3]
        $x = $g->Variable($K->array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]));
        $z = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x) {
                $z = $g->reshape($x,[1,2,3]);
                return $z;
            }
        );

        $this->assertEquals("[[1,2],[3,4],[5,6]]",$mo->toString($K->ndarray($x->value())));
        $this->assertEquals("[[[1,2,3],[4,5,6]]]",$mo->toString($K->ndarray($z->value())));
        $this->assertTrue($x->isbackpropagatable());
        $this->assertTrue($z->isbackpropagatable());
        $dx = $tape->gradient($z,$x);
        $this->assertEquals("[[1,1],[1,1],[1,1]]",$mo->toString($K->ndarray($dx)));
    }

    public function testBatchSize()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        // reshape [2,1,3] => [2,3]
        $x = $g->Variable($K->array([ [[1.0,2.0,3.0]], [[4.0,5.0,6.0]] ]));
        $this->assertEquals([2,1,3],$x->shape());
        $z = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x) {
                $z = $g->reshape($x,[0,3]); // dim 0 means batchsize
                return $z;
            }
        );
        $this->assertEquals([2,3],$z->shape());

        $this->assertEquals("[[[1,2,3]],[[4,5,6]]]",$mo->toString($K->ndarray($x->value())));
        $this->assertEquals("[[1,2,3],[4,5,6]]",$mo->toString($K->ndarray($z->value())));
        $this->assertTrue($x->isbackpropagatable());
        $this->assertTrue($z->isbackpropagatable());
        $dx = $tape->gradient($z,$x);
        $this->assertEquals("[[[1,1,1]],[[1,1,1]]]",$mo->toString($K->ndarray($dx)));
    }

    public function testFitShapeNormal()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        // reshape [2,3,4] => [2,2,6]
        $x = $g->Variable($K->ones([2,3,4]));
        $this->assertEquals([2,3,4],$x->shape());
        $z = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x) {
                $z = $g->reshape($x,[0,2,-1]); // dim -1 means fit shapesize
                return $z;
            }
        );
        $this->assertEquals([2,2,6],$z->shape());

        $this->assertTrue($x->isbackpropagatable());
        $this->assertTrue($z->isbackpropagatable());
        $dx = $tape->gradient($z,$x);
        $this->assertEquals([2,3,4],$dx->shape());
    }

    public function testFitShapeDuplicate()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $this->expectException(InvalidArgumentException::class);

        // reshape [2,1,3] => [2,3]
        $x = $g->Variable($K->array([ [[1.0,2.0,3.0]], [[4.0,5.0,6.0]] ]));
        $this->assertEquals([2,1,3],$x->shape());
        $z = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x) {
                $z = $g->reshape($x,[0,-1,-1]); // dim -1 means fit shapesize
                return $z;
            }
        );
    }
}
