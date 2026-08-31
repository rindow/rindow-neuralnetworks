<?php
namespace RindowTest\NeuralNetworks\Gradient\Func\SqueezeTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

class SqueezeTest extends TestCase
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

    public function testMiddleAxisNormal()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();


        $inputs = $g->Variable($K->zeros([2,1,3]));
        $copyInputs = $K->copy($inputs);
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$inputs) {
                $outputsVariable = $g->squeeze($inputs,1);
                return $outputsVariable;
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        //
        $this->assertEquals([2,1,3],$inputs->shape());
        $this->assertEquals([2,3],$outputs->shape());
        $this->assertEquals($copyInputs->toArray(),$inputs->toArray());

        $dParams = $tape->gradient($outputsVariable,$inputs);

        $this->assertEquals([2,1,3],$dParams->shape());
    }

    public function testLeftAxisNormal()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();


        $inputs = $g->Variable($K->zeros([1,2,3]));
        $copyInputs = $K->copy($inputs);
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$inputs) {
                $outputsVariable = $g->squeeze($inputs,0);
                return $outputsVariable;
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        //
        $this->assertEquals([1,2,3],$inputs->shape());
        $this->assertEquals([2,3],$outputs->shape());
        $this->assertEquals($copyInputs->toArray(),$inputs->toArray());

        $dParams = $tape->gradient($outputsVariable,$inputs);

        $this->assertEquals([1,2,3],$dParams->shape());
    }

    public function testRightAxisNormal()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();


        $inputs = $g->Variable($K->zeros([2,3,1]));
        $copyInputs = $K->copy($inputs);
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$inputs) {
                $outputsVariable = $g->squeeze($inputs,-1);
                return $outputsVariable;
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        //
        $this->assertEquals([2,3,1],$inputs->shape());
        $this->assertEquals([2,3],$outputs->shape());
        $this->assertEquals($copyInputs->toArray(),$inputs->toArray());

        $dParams = $tape->gradient($outputsVariable,$inputs);

        $this->assertEquals([2,3,1],$dParams->shape());
    }

}
