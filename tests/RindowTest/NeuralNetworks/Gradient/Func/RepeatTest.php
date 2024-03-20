<?php
namespace RindowTest\NeuralNetworks\Gradient\Func\RepeatTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;

class RepeatTest extends TestCase
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

    public function testNormal()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $func = $g->Function(
            function($x,$repeats,$b) use ($g){
                $y = $g->repeat($x,$repeats,axis:-1);
                $y = $g->mul($y,$b);
                return $y;
            }
        );

        // build
        $x = $g->Variable($K->array($mo->arange(4,dtype:NDArray::float32)->reshape([2,2])));
        $b = $g->Variable($K->array($mo->arange(12,dtype:NDArray::float32)->reshape([2,3,2])));
        $repeats = $g->Variable($K->array(3,dtype:NDArray::int32));
        $y = $nn->with($tape=$g->GradientTape(),$func,[$x,$repeats,$b],true);
        $grads = $tape->gradient($y,[$x]);
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->multiply(
                $mo->la()->array([[[0,1],[0,1],[0,1]],[[2,3],[2,3],[2,3]]]),  // repeat(x)
                $mo->la()->array([[[0,1],[2,3],[4,5]],[[6,7],[8,9],[10,11]]]) // b
            ),
            $K->ndarray($y->value())));
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->array([[6,9],[24,27]]),
            $K->ndarray($grads[0])));
        
        // exec
        $x = $g->Variable($K->array($mo->arange(4,dtype:NDArray::float32)->reshape([2,2])));
        $b = $g->Variable($K->array($mo->arange(12,dtype:NDArray::float32)->reshape([2,3,2])));
        $repeats = $g->Variable($K->array(3,dtype:NDArray::int32));
        $y = $nn->with($tape=$g->GradientTape(),$func,[$x,$repeats,$b],true);
        $grads = $tape->gradient($y,[$x]);
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->multiply(
                $mo->la()->array([[[0,1],[0,1],[0,1]],[[2,3],[2,3],[2,3]]]),  // repeat(x)
                $mo->la()->array([[[0,1],[2,3],[4,5]],[[6,7],[8,9],[10,11]]]) // b
            ),
            $K->ndarray($y->value())));
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->array([[6,9],[24,27]]),
            $K->ndarray($grads[0])));
    }

    public function testKeepdims()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $func = $g->Function(
            function($x,$repeats,$b) use ($g){
                $y = $g->repeat($x,$repeats,axis:-1,keepdims:true);
                $y = $g->mul($y,$b);
                return $y;
            }
        );

        // build
        $x = $g->Variable($K->array($mo->arange(4,dtype:NDArray::float32)->reshape([2,2])));
        $b = $g->Variable($K->array($mo->arange(12,dtype:NDArray::float32)->reshape([2,6])));
        $repeats = $g->Variable($K->array(3,dtype:NDArray::int32));
        $y = $nn->with($tape=$g->GradientTape(),$func,[$x,$repeats,$b],true);
        $grads = $tape->gradient($y,[$x]);
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->multiply(
                $mo->la()->array([[0,1,0,1,0,1],[2,3,2,3,2,3]]),  // repeat(x)
                $mo->la()->array([[0,1,2,3,4,5],[6,7,8,9,10,11]]) // b
            ),
            $K->ndarray($y->value())));
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->array([[6,9],[24,27]]),
            $K->ndarray($grads[0])));
        
        // exec
        $x = $g->Variable($K->array($mo->arange(4,dtype:NDArray::float32)->reshape([2,2])));
        $b = $g->Variable($K->array($mo->arange(12,dtype:NDArray::float32)->reshape([2,6])));
        $repeats = $g->Variable($K->array(3,dtype:NDArray::int32));
        $y = $nn->with($tape=$g->GradientTape(),$func,[$x,$repeats,$b],true);
        $grads = $tape->gradient($y,[$x]);
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->multiply(
                $mo->la()->array([[0,1,0,1,0,1],[2,3,2,3,2,3]]),  // repeat(x)
                $mo->la()->array([[0,1,2,3,4,5],[6,7,8,9,10,11]]) // b
            ),
            $K->ndarray($y->value())));
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->array([[6,9],[24,27]]),
            $K->ndarray($grads[0])));
    }
}
