<?php
namespace RindowTest\NeuralNetworks\Layer\DenseTest;

use InvalidArgumentException;
use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Layer\Dense;

class Test extends TestCase
{
    public function newMatrixOperator()
    {
        return new MatrixOperator();
    }

    public function newNeuralNetworks($mo)
    {
        return new NeuralNetworks($mo);
    }

    public function verifyGradient($mo, $nn, $K, $g, $function, NDArray $x)
    {
        $f = function($x) use ($mo,$K,$function){
            $x = $K->array($x);
            $y = $function->forward($x,$training=true);
            return $K->ndarray($y);
        };
        $grads = $mo->la()->numericalGradient(1e-3,$f,$K->ndarray($x));

        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($function,$x) {
                $outputsVariable = $function->forward($x, $training=true);
                return $outputsVariable;
            }
        );
        $dOutputs = $K->ones($outputsVariable->shape(),$outputsVariable->dtype());
        $dInputs = $outputsVariable->creator()->backward([$dOutputs]);

        return $mo->la()->isclose($grads[0],$K->ndarray($dInputs[0]),1e-3);
    }

    public function testDefaultInitialize()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $layer = new Dense($K,$units=3,input_shape:[2]);

        $layer->build();
        $params = $layer->getParams();
        $this->assertCount(2,$params);
        $this->assertEquals([2,3],$params[0]->shape());
        $this->assertEquals([3],$params[1]->shape());
        $this->assertNotEquals($mo->zeros([2,3])->toArray(),$params[0]->toArray());
        $this->assertEquals($mo->zeros([3])->toArray(),$params[1]->toArray());

        $grads = $layer->getGrads();
        $this->assertCount(2,$grads);
        $this->assertEquals([2,3],$grads[0]->shape());
        $this->assertEquals([3],$grads[1]->shape());
        $this->assertEquals($mo->zeros([2,3])->toArray(),$grads[0]->toArray());
        $this->assertEquals($mo->zeros([3])->toArray(),$grads[1]->toArray());

        $this->assertEquals([3],$layer->outputShape());
        //$layer->unlink();
    }

    public function testSetInputShape()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new Dense($K,$units=3);
        $inputs = $g->Variable($K->zeros([1,2]));
        $layer->build($inputs);
        $params = $layer->getParams();
        $this->assertCount(2,$params);
        $this->assertEquals([2,3],$params[0]->shape());

        $this->assertEquals([3],$layer->outputShape());
    }

    public function testUnmatchSpecifiedInputShape()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new Dense($K,$units=3,input_shape:[2]);
    
        $inputs = $g->Variable($K->zeros([1,3]));
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Input shape is inconsistent: defined as [2] but [3] given in Dense');
        $layer->build($inputs);
    }

    public function testNormalForwardAndBackward()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $fn = $mo->la();

        $layer = new Dense($K,$units=2,input_shape:[3]);

        // 3 input x 4 minibatch
        $inputs = $K->array([
            [0.0, 0.0 , 6.0],
            [0.0, 0.0 , 6.0],
            [0.0, 0.0 , 6.0],
            [0.0, 0.0 , 6.0],
        ]);

        $layer->build($g->Variable($inputs),
            sampleWeights:[
                $K->array([[0.1, 0.2], [0.1, 0.1], [0.2, 0.2]]), // kernel
                $K->array([0.5, 0.1]),                           // bias
            ]
        );

        //
        // forward
        //
        $copyInputs = $K->copy($inputs);
        $inputs = $K->array($inputs);
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$inputs) {
                $outputsVariable = $layer->forward($inputs, $training=true);
                return $outputsVariable;
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        $inputs = $K->ndarray($inputs);
        // 2 output x 4 batch
        $this->assertEquals([4,2],$outputs->shape());
        $this->assertTrue($fn->isclose($mo->array([
                [1.7, 1.3],
                [1.7, 1.3],
                [1.7, 1.3],
                [1.7, 1.3],
            ]),$outputs
        ));
        $this->assertEquals($copyInputs->toArray(),$inputs->toArray());

        //
        // backward
        //
        // 2 output x 4 batch
        $dOutputs = $mo->array([
            [0.0, -0.5],
            [0.0, -0.5],
            [0.0, -0.5],
            [0.0, -0.5],
        ]);
        $copydOutputs = $mo->copy($dOutputs);
        $dOutputs = $K->array($dOutputs);
        [$dInputs] = $outputsVariable->creator()->backward([$dOutputs]);
        $dInputs = $K->ndarray($dInputs);
        $dOutputs = $K->ndarray($dOutputs);
        // 3 input x 4 batch
        $this->assertEquals([4,3],$dInputs->shape());
        $this->assertTrue($fn->isclose($mo->array([
                [-0.1, -0.05 , -0.1],
                [-0.1, -0.05 , -0.1],
                [-0.1, -0.05 , -0.1],
                [-0.1, -0.05 , -0.1],
            ]),$dInputs
        ));
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());
    }

    public function testNdInput()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $layer = new Dense($K,$units=4,input_shape:[2,3]);

        $layer->build();
        $params = $layer->getParams();
        $this->assertCount(2,$params);
        $this->assertEquals([3,4],$params[0]->shape());
        $this->assertEquals([4],$params[1]->shape());
        $this->assertNotEquals($mo->zeros([2,4])->toArray(),$params[0]->toArray());
        $this->assertEquals($mo->zeros([4])->toArray(),$params[1]->toArray());

        $grads = $layer->getGrads();
        $this->assertCount(2,$grads);
        $this->assertEquals([3,4],$grads[0]->shape());
        $this->assertEquals([4],$grads[1]->shape());
        $this->assertEquals($mo->zeros([3,4])->toArray(),$grads[0]->toArray());
        $this->assertEquals($mo->zeros([4])->toArray(),$grads[1]->toArray());

        $this->assertEquals([2,4],$layer->outputShape());

        $inputs = $mo->zeros([10,2,3]);
        $inputs = $K->array($inputs);
        $outputs = $layer->forward($inputs,true);
        $outputs = $K->ndarray($outputs);
        $this->assertEquals([10,2,4],$outputs->shape());
    }

    public function testGradientWithActivation()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $fn = $mo->la();

        $layer = new Dense($K,$units=2,input_shape:[3],activation:'tanh');

        // 3 input x 4 minibatch
        $inputs = $K->ones([4,3]);

        $layer->build($g->Variable($inputs));

        //
        // forward
        //
        $copyInputs = $K->copy($inputs);
        $inputs = $K->array($inputs);
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$inputs) {
                $outputsVariable = $layer->forward($inputs, $training=true);
                return $outputsVariable;
            }
        );

        $this->assertTrue(
            $this->verifyGradient($mo,$nn,$K,$g,$layer,$inputs));
    }
}
