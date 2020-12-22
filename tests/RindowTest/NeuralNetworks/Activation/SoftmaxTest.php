<?php
namespace RindowTest\NeuralNetworks\Activation\SoftmaxTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Activation\Softmax;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

class Test extends TestCase
{
    public function newMatrixOperator()
    {
        return new MatrixOperator();
    }

    public function newBackend($mo)
    {
        $builder = new NeuralNetworks($mo);
        return $builder->backend();
    }

    public function verifyGradient($mo, $K, $function, NDArray $x, ...$args)
    {
        $f = function($x) use ($K,$function,$args){
            $x = $K->array($x);
            $y = $function->forward($x,...$args);
            return $K->ndarray($y);
        };
        $grads = $mo->la()->numericalGradient(1e-3,$f,$x);
        $outputs = $K->ndarray($function->forward($K->array($x), ...$args));
        $ones = $mo->ones($outputs->shape(),$outputs->dtype());
        $dInputs = $K->ndarray($function->backward($K->array($ones)));
        return $mo->la()->isclose($grads[0],$dInputs,null,1e-4);
    }

    public function testNormal()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $activation = new Softmax($K);

        $x = $mo->array([
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
        ]);
        $copyX = $mo->copy($x);
        $x = $K->array($x);
        $y = $activation->forward($x, $training=true);
        $y = $K->ndarray($y);
        $x = $K->ndarray($x);
        $this->assertEquals($copyX->toArray(),$x->toArray());
        $this->assertEquals($x->shape(),$y->shape());
        $this->assertTrue($mo->la()->isclose(
            $mo->ones([3]),
            $mo->sum($y,$axis=1)
        ));

        $dout = $mo->array([
            [-0.5,-0.25,0.0,0.25,0.5],
            [-0.5,-0.25,0.0,0.25,0.5],
            [-0.5,-0.25,0.0,0.25,0.5],
        ]);
        $copydout = $mo->copy($dout);
        $dout = $K->array($dout);
        $dx = $activation->backward($dout);
        $dx = $K->ndarray($dx);
        $dout = $K->ndarray($dout);
        $this->assertEquals($dout->shape(),$dx->shape());
        $this->assertEquals($copydout->toArray(),$dout->toArray());

        $inputs = $mo->array([
            [-20.0,-15.0,0.0,5.0,10.0],
            [-10.0,-0.5,0.0,0.5,10.0],
            [-10.0,-0.5,0.0,0.5,10.0],
        ]);
        $this->assertTrue(
            $this->verifyGradient($mo,$K,$activation,$inputs,$training=true));
    }
}
