<?php
namespace RindowTest\NeuralNetworks\Activation\TanhTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Activation\Tanh;

class Test extends TestCase
{
    public function verifyGradient($mo, $function, NDArray $x, ...$args)
    {
        $f = function($x) use ($function,$args){
            return $function->forward($x,...$args);
        };
        $grads = $mo->la()->numericalGradient(null,$f,$x);
        $outputs = $function->forward($x, ...$args);
        $ones = $mo->ones($outputs->shape(),$outputs->dtype());
        $dInputs = $function->backward($ones);
        return $mo->la()->isclose($grads[0],$dInputs);
    }

    public function testNormal()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $activation = new Tanh($backend);

        $inputs = $mo->array([
            [-1.0,-0.5,0.0,0.5,1.0],
            [-2.0,-1.0,0.0,1.0,2.0],
        ]);
        $copyInputs = $mo->copy($inputs);
        $outputs = $activation->forward($inputs, $training=true);
        $this->assertEquals([2,5],$outputs->shape());
        $this->assertEquals($copyInputs->toArray(),$inputs->toArray());

        $dOutputs = $mo->array([
            [-1.0,-0.5,0.0,0.5,1.0],
            [-2.0,-1.0,0.0,1.0,2.0],
        ]);
        $copydOutputs = $mo->copy($dOutputs);
        $dInputs = $activation->backward($dOutputs);
        $this->assertEquals([2,5],$dInputs->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());

        $this->assertTrue(
            $this->verifyGradient($mo,$activation,$inputs,$training=true));
    }
}
