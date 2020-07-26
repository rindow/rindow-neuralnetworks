<?php
namespace RindowTest\NeuralNetworks\Layer\Conv3DTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Layer\Conv3D;
use InvalidArgumentException;

class Test extends TestCase
{
    public function testDefaultInitialize()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $layer = new Conv3D(
            $backend,
            $filters=5,
            $kernel_size=3,
            [
                'input_shape'=>[4,4,4,1]
            ]);

        $layer->build();
        $params = $layer->getParams();
        $this->assertCount(2,$params);
        $this->assertEquals([3,3,3,1,5],$params[0]->shape());
        $this->assertEquals([5],$params[1]->shape());
        $this->assertNotEquals($mo->zeros([3,3,3,1,5])->toArray(),$params[0]->toArray());
        $this->assertEquals($mo->zeros([5])->toArray(),$params[1]->toArray());

        $grads = $layer->getGrads();
        $this->assertCount(2,$grads);
        $this->assertEquals([3,3,3,1,5],$grads[0]->shape());
        $this->assertEquals([5],$grads[1]->shape());
        $this->assertEquals($mo->zeros([3,3,3,1,5])->toArray(),$grads[0]->toArray());
        $this->assertEquals($mo->zeros([5])->toArray(),$grads[1]->toArray());

        $this->assertEquals([2,2,2,5],$layer->outputShape());
    }

    public function testNotspecifiedInputShape()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $layer = new Conv3D(
            $backend,
            $filters=5,
            $kernel_size=3,
            [
            ]);

        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Input shape is not defined');
        $layer->build();
    }

    public function testSetInputShape()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $layer = new Conv3D(
            $backend,
            $filters=5,
            $kernel_size=3,
            [
            ]);
        $layer->build($inputShape=[4,4,4,1]);
        $params = $layer->getParams();
        $this->assertCount(2,$params);
        $this->assertEquals([3,3,3,1,5],$params[0]->shape());

        $this->assertEquals([2,2,2,5],$layer->outputShape());
    }

    public function testNormalForwardAndBackword()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $fn = $backend;

        $layer = new Conv3D(
            $backend,
            $filters=2,
            $kernel_size=2,
            ['input_shape'=>[3,3,3,1]]);

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
            ['sampleWeights'=>
                [$kernel,$bias]
        ]);*/
        $layer->build();
        [$kernel,$bias]=$layer->getParams();
        $this->assertEquals(
            [2,2,2,1,2],
            $kernel->shape());
        $this->assertEquals(
            [2],
            $bias->shape());

        //
        // forward
        //
        //  batch size 2
        $inputs = $mo->ones([2,3,3,3,1]);
        $this->assertEquals(
            [2,3,3,3,1],
            $inputs->shape());
        $copyInputs = $mo->copy($inputs);
        $outputs = $layer->forward($inputs, $training=true);
        // 
        $this->assertEquals(
            [2,2,2,2,2],$outputs->shape());
        $this->assertEquals($copyInputs->toArray(),$inputs->toArray());

        //
        // backword
        //
        // 2 batch
        $dOutputs = $mo->full([2,2,2,2,2],0.1);
        $copydOutputs = $mo->copy(
            $dOutputs);
        $dInputs = $layer->backward($dOutputs);
        // 2 batch
        $this->assertEquals([2,3,3,3,1],$dInputs->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());
    }

}
