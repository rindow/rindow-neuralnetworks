<?php
namespace RindowTest\NeuralNetworks\Layer\MaxPooling2DTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Layer\MaxPooling2D;
use InvalidArgumentException;

class Test extends TestCase
{
    public function testDefaultInitialize()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $layer = new MaxPooling2D(
            $backend,
            [
                'input_shape'=>[4,4,3]
            ]);

        $layer->build();
        $params = $layer->getParams();
        $this->assertCount(0,$params);

        $grads = $layer->getGrads();
        $this->assertCount(0,$grads);

        $this->assertEquals([2,2,3],$layer->outputShape());
    }

    public function testNotspecifiedInputShape()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $layer = new MaxPooling2D(
            $backend,
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
        $layer = new MaxPooling2D(
            $backend,
            [
            ]);
        $layer->build($inputShape=[4,4,3]);

        $this->assertEquals([2,2,3],$layer->outputShape());
    }

    public function testNormalForwardAndBackward()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $fn = $backend;

        $layer = new MaxPooling2D(
            $backend,
            ['input_shape'=>[4,4,3]]);

        $layer->build();

        //
        // forward
        //
        //  batch size 2
        $inputs = $mo->ones([2,4,4,3]);
        $copyInputs = $mo->copy($inputs);
        $outputs = $layer->forward($inputs, $training=true);
        //
        $this->assertEquals(
            [2,2,2,3],$outputs->shape());
        $this->assertEquals($copyInputs->toArray(),$inputs->toArray());

        //
        // backward
        //
        // 2 batch
        $dOutputs = $mo->op(
            $mo->ones([2,2,2,3]),
            '*',
            0.1);

        $copydOutputs = $mo->copy(
            $dOutputs);
        $dInputs = $layer->backward($dOutputs);
        // 2 batch
        $this->assertEquals([2,4,4,3],$dInputs->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());
    }
}
