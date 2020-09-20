<?php
namespace RindowTest\NeuralNetworks\Activation\SigmoidTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Activation\Sigmoid;


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
        return $mo->la()->isclose($grads[0],$dInputs);
    }

    public function testNormal()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $activation = new Sigmoid($backend);

        $x = $mo->array([-1.0,-0.5,0.0,0.5,1.0]);
        $y = $activation->forward($x, $training=true);

        $this->assertEquals([-1.0,-0.5,0.0,0.5,1.0],$x->toArray());
        $this->assertTrue($y[0]<0.5);
        $this->assertTrue($y[1]<0.5);
        $this->assertTrue($y[2]==0.5);
        $this->assertTrue($y[3]>0.5);
        $this->assertTrue($y[4]>0.5);

        $dout = $x;
        $dx = $activation->backward($dout);
        $this->assertTrue(abs(-0.196-$dx[0])<0.01);
        $this->assertTrue(abs(-0.117-$dx[1])<0.01);
        $this->assertTrue($dx[2]==0.0);
        $this->assertTrue(abs(0.117-$dx[3])<0.01);
        $this->assertTrue(abs(0.196-$dx[4])<0.01);

        $inputs = $mo->array([-1.0,-0.5,0.0,0.5,1.0]);
        $this->assertTrue(
            $this->verifyGradient($mo,$activation,$inputs,$training=true));
    }
}
