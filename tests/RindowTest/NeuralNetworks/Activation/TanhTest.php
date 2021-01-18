<?php
namespace RindowTest\NeuralNetworks\Activation\TanhTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Activation\Tanh;
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
        $grads = $mo->la()->numericalGradient(null,$f,$K->ndarray($x));
        $outputs = $function->forward($x, ...$args);
        $ones = $K->ones($outputs->shape(),$outputs->dtype());
        $dInputs = $function->backward($ones);
        return $mo->la()->isclose($grads[0],$K->ndarray($dInputs));
    }

    public function testNormal()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $activation = new Tanh($K);

        $inputs = $K->array([
            [-1.0,-0.5,0.0,0.5,1.0],
            [-2.0,-1.0,0.0,1.0,2.0],
        ]);
        $copyInputs = $K->copy($inputs);
        $inputs = $K->array($inputs);
        $outputs = $activation->forward($inputs, $training=true);
        $this->assertEquals([2,5],$outputs->shape());
        $this->assertEquals($copyInputs->toArray(),$inputs->toArray());

        $dOutputs = $K->array([
            [-1.0,-0.5,0.0,0.5,1.0],
            [-2.0,-1.0,0.0,1.0,2.0],
        ]);
        $copydOutputs = $K->copy($dOutputs);
        $dOutputs = $K->array($dOutputs);
        $dInputs = $activation->backward($dOutputs);
        $this->assertEquals([2,5],$dInputs->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());

        $this->assertTrue(
            $this->verifyGradient($mo,$K,$activation,$inputs,$training=true));
    }
}
