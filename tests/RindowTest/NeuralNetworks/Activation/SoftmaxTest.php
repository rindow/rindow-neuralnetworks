<?php
namespace RindowTest\NeuralNetworks\Activation\SoftmaxTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Activation\Softmax;


class Test extends TestCase
{
    public function verifyGradient($mo, $function, NDArray $x, ...$args)
    {
        $f = function($x) use ($function,$args){
            return $function->forward($x,...$args);
        };
        $grads = $mo->la()->numericalGradient(1e-3,$f,$x);
        $outputs = $function->forward($x, ...$args);
        $ones = $mo->ones($outputs->shape(),$outputs->dtype());
        $dInputs = $function->backward($ones);
        return $mo->la()->isclose($grads[0],$dInputs,null,1e-4);
    }

    public function testNormal()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $activation = new Softmax($backend);

        $x = $mo->array([
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
        ]);
        $copyX = $mo->copy($x);
        $y = $activation->forward($x, $training=true);
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
        $dx = $activation->backward($dout);
        $this->assertEquals($dout->shape(),$dx->shape());
        $this->assertEquals($copydout->toArray(),$dout->toArray());

        $inputs = $mo->array([
            [-20.0,-15.0,0.0,5.0,10.0],
            [-10.0,-0.5,0.0,0.5,10.0],
            [-10.0,-0.5,0.0,0.5,10.0],
        ]);
        $this->assertTrue(
            $this->verifyGradient($mo,$activation,$inputs,$training=true));
    }
}
