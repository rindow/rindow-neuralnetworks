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
            $states = new \stdClass();
            $x = $K->array($x);
            $y = $function->forward($states,$x,...$args);
            return $K->ndarray($y);
        };
        $grads = $mo->la()->numericalGradient(1e-3,$f,$K->ndarray($x));
        $states = new \stdClass();
        $outputs = $function->forward($states,$x, ...$args);
        $ones = $K->ones($outputs->shape(),$outputs->dtype());
        $dInputs = $function->backward($states,$ones);

        //echo $mo->toString($K->sub($grads[0],$K->ndarray($dInputs)),'%6.4e',true)."\n";
        return $mo->la()->isclose($grads[0],$K->ndarray($dInputs));
    }

    public function testNormal()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $activation = new Tanh($K);

        $states = new \stdClass();
        $inputs = $K->array([
            [-1.0,-0.5,0.0,0.5,1.0],
            [-2.0,-1.0,0.0,1.0,2.0],
        ]);
        $inputs = $K->scale(1/1,$inputs);
        $copyInputs = $K->copy($inputs);
        $inputs = $K->array($inputs);
        $outputs = $activation->forward($states,$inputs, $training=true);
        $this->assertEquals([2,5],$outputs->shape());
        $this->assertEquals($copyInputs->toArray(),$inputs->toArray());

        $dOutputs = $K->array([
            [-1.0,-0.5,0.0,0.5,1.0],
            [-2.0,-1.0,0.0,1.0,2.0],
        ]);
        $copydOutputs = $K->copy($dOutputs);
        $dOutputs = $K->array($dOutputs);
        $dInputs = $activation->backward($states,$dOutputs);
        $this->assertEquals([2,5],$dInputs->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());

        $this->assertTrue(
            $this->verifyGradient($mo,$K,$activation,$inputs,$training=true));
    }
}
