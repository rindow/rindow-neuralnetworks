<?php
namespace RindowTest\NeuralNetworks\Activation\SigmoidTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Activation\Sigmoid;
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
        $grads = $mo->la()->numericalGradient(1e-3,$f,$K->ndarray($x));
        $outputs = $function->forward($x, ...$args);
        $ones = $K->ones($outputs->shape(),$outputs->dtype());
        $dInputs = $function->backward($ones);
        return $mo->la()->isclose($grads[0],$K->ndarray($dInputs));
    }

    public function testNormal()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $activation = new Sigmoid($K);

        $x = $mo->array([-1.0,-0.5,0.0,0.5,1.0]);
        $x = $K->array($x);
        $y = $activation->forward($x, $training=true);
        $y = $K->ndarray($y);
        $x = $K->ndarray($x);

        $this->assertEquals([-1.0,-0.5,0.0,0.5,1.0],$x->toArray());
        $this->assertTrue($y[0]<0.5);
        $this->assertTrue($y[1]<0.5);
        $this->assertTrue($y[2]==0.5);
        $this->assertTrue($y[3]>0.5);
        $this->assertTrue($y[4]>0.5);

        $dout = $x;
        $copydout = $mo->copy($dout);
        $dout = $K->array($dout);
        $dx = $activation->backward($dout);
        $dx = $K->ndarray($dx);
        $dout = $K->ndarray($dout);
        $this->assertTrue(abs(-0.196-$dx[0])<0.01);
        $this->assertTrue(abs(-0.117-$dx[1])<0.01);
        $this->assertTrue($dx[2]==0.0);
        $this->assertTrue(abs(0.117-$dx[3])<0.01);
        $this->assertTrue(abs(0.196-$dx[4])<0.01);
        $this->assertEquals($copydout->toArray(),$dout->toArray());

        $inputs = $K->array([-1.0,-0.5,0.0,0.5,1.0]);
        $this->assertTrue(
            $this->verifyGradient($mo,$K,$activation,$inputs,$training=true));
    }
}
