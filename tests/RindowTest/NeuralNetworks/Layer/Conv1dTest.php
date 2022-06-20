<?php
namespace RindowTest\NeuralNetworks\Layer\Conv1DTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Layer\Conv1D;
use InvalidArgumentException;

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

    public function testDefaultInitialize()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();

        $layer = new Conv1D(
            $K,
            $filters=5,
            $kernel_size=3,
            input_shape:[4,1]
            );

        $inputs = $g->Variable($K->zeros([1,4,1]));
        $layer->build($inputs);
        $params = $layer->getParams();
        $this->assertCount(2,$params);
        $this->assertEquals([3,1,5],$params[0]->shape());
        $this->assertEquals([5],$params[1]->shape());
        $this->assertNotEquals($mo->zeros([3,1,5])->toArray(),$params[0]->toArray());
        $this->assertEquals($mo->zeros([5])->toArray(),$params[1]->toArray());

        $grads = $layer->getGrads();
        $this->assertCount(2,$grads);
        $this->assertEquals([3,1,5],$grads[0]->shape());
        $this->assertEquals([5],$grads[1]->shape());
        $this->assertEquals($mo->zeros([3,1,5])->toArray(),$grads[0]->toArray());
        $this->assertEquals($mo->zeros([5])->toArray(),$grads[1]->toArray());

        $this->assertEquals([2,5],$layer->outputShape());
    }

    public function testSetInputShape()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new Conv1D(
            $K,
            $filters=5,
            $kernel_size=3,
            );
        $inputs = $g->Variable($K->zeros([1,4,1]));
        $layer->build($inputs);
        $params = $layer->getParams();
        $this->assertCount(2,$params);
        $this->assertEquals([3,1,5],$params[0]->shape());

        $this->assertEquals([2,5],$layer->outputShape());
    }

    public function testUnmatchSpecifiedInputShape()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new Conv1D($K,
            $filters=5,
            $kernel_size=3,
            input_shape:[4,1]);
        $inputs = $g->Variable($K->zeros([1,4,2]));
    
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Input shape is inconsistent: defined as [4,1] but [4,2] given in Conv1D');
        $layer->build($inputs);
    }

    public function testNormalForwardAndBackward()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();

        $layer = new Conv1D(
            $K,
            $filters=2,
            $kernel_size=2,
            input_shape:[3,1]);

        //  batch size 2
        $inputs = $K->array([
            [[0.0],[0.0],[6.0]],
            [[0.0],[0.0],[6.0]],
          ]);
  
        /*
        $kernel = $mo->array([
               [[[0.1, 0.2]],
                [[0.1, 0.1]]],
               [[[0.2, 0.2]],
                [[0.2, 0.1]]]
            ]); // kernel
        $bias = $mo->array(
                [0.5,0.1]
            );  // bias
        $layer->build(null,
             sampleWeights:[$kernel,$bias]
        );*/
        $layer->build($g->Variable($inputs));
        [$kernel,$bias]=$layer->getParams();
        $this->assertEquals(
            [2,1,2],
            $kernel->shape());
        $this->assertEquals(
            [2],
            $bias->shape());

        //
        // forward
        //
        $this->assertEquals(
            [2,3,1],
            $inputs->shape());
        $copyInputs = $K->copy($inputs);
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$inputs) {
                $outputsVariable = $layer->forward($inputs, $training=true);
                return $outputsVariable;
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        //
        $this->assertEquals(
            [2,2,2],$outputs->shape());
        $this->assertEquals($copyInputs->toArray(),$inputs->toArray());

        //
        // backward
        //
        // 2 batch
        $dOutputs = $K->array([
               [[0, 0.1],
                [0,-0.1]],
               [[0, 0.1],
                [0,-0.1]],
            ]);
        $copydOutputs = $K->copy(
            $dOutputs);
        [$dInputs] = $outputsVariable->creator()->backward([$dOutputs]);
        // 2 batch
        $this->assertEquals([2,3,1],$dInputs->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());
    }
}
