<?php
namespace RindowTest\NeuralNetworks\Layer\DenseTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Layer\Dense;
use InvalidArgumentException;

class Test extends TestCase
{
    public function testDefaultInitialize()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $layer = new Dense($backend,$units=3,['input_shape'=>[2]]);

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
    }

    public function testNotspecifiedInputShape()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $layer = new Dense($backend,$units=3,[]);

        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Input shape is not defined');
        $layer->build();
    }

    public function testSetInputShape()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $layer = new Dense($backend,$units=3,[]);
        $layer->build($inputShape=[2]);
        $params = $layer->getParams();
        $this->assertCount(2,$params);
        $this->assertEquals([2,3],$params[0]->shape());

        $this->assertEquals([3],$layer->outputShape());
    }

    public function testNormalForwardAndBackward()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $fn = $backend;

        $layer = new Dense($backend,$units=2,['input_shape'=>[3]]);

        $layer->build(null,[
            'sampleWeights'=>[
                $mo->array([[0.1, 0.2], [0.1, 0.1], [0.2, 0.2]]), // kernel
                $mo->array([0.5, 0.1]),                         // bias
            ]
        ]);

        //
        // forward
        //
        // 3 input x 4 minibatch
        $inputs = $mo->array([
            [0.0, 0.0 , 6.0],
            [0.0, 0.0 , 6.0],
            [0.0, 0.0 , 6.0],
            [0.0, 0.0 , 6.0],
        ]);
        $copyInputs = $mo->copy($inputs);
        $outputs = $layer->forward($inputs, $training=true);
        // 2 output x 4 batch
        $this->assertEquals([4,2],$outputs->shape());
        $this->assertTrue($fn->equalTest($mo->array([
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
        $dInputs = $layer->backward($dOutputs);
        // 3 input x 4 batch
        $this->assertEquals([4,3],$dInputs->shape());
        $this->assertTrue($fn->equalTest($mo->array([
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
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $layer = new Dense($backend,$units=4,['input_shape'=>[2,3]]);

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
        $outputs = $layer->forward($inputs,true);
        $this->assertEquals([10,2,4],$outputs->shape());
    }
}
