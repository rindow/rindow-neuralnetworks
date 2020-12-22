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
        $grads = $mo->la()->numericalGradient(null,$f,$x);
        $outputs = $K->ndarray($function->forward($K->array($x), ...$args));
        $ones = $mo->ones($outputs->shape(),$outputs->dtype());
        $dInputs = $K->ndarray($function->backward($K->array($ones)));
        return $mo->la()->isclose($grads[0],$dInputs);
    }

    public function testNormal()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $activation = new Tanh($K);

        $inputs = $mo->array([
            [-1.0,-0.5,0.0,0.5,1.0],
            [-2.0,-1.0,0.0,1.0,2.0],
        ]);
        $copyInputs = $mo->copy($inputs);
        $inputs = $K->array($inputs);
        $outputs = $activation->forward($inputs, $training=true);
        $outputs = $K->ndarray($outputs);
        $inputs = $K->ndarray($inputs);
        $this->assertEquals([2,5],$outputs->shape());
        $this->assertEquals($copyInputs->toArray(),$inputs->toArray());

        $dOutputs = $mo->array([
            [-1.0,-0.5,0.0,0.5,1.0],
            [-2.0,-1.0,0.0,1.0,2.0],
        ]);
        $copydOutputs = $mo->copy($dOutputs);
        $dOutputs = $K->array($dOutputs);
        $dInputs = $activation->backward($dOutputs);
        $dInputs = $K->ndarray($dInputs);
        $dOutputs = $K->ndarray($dOutputs);
        $this->assertEquals([2,5],$dInputs->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());

        $this->assertTrue(
            $this->verifyGradient($mo,$K,$activation,$inputs,$training=true));
    }
}
