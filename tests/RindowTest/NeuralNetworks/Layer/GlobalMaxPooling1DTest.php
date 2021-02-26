<?php
namespace RindowTest\NeuralNetworks\Layer\GlobalMaxPooling1DTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Layer\GlobalMaxPooling1D;
use InvalidArgumentException;

class Test extends TestCase
{
    public function newBackend($mo)
    {
        $builder = new NeuralNetworks($mo);
        return $builder->backend();
    }

    public function testDefaultInitialize()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $layer = new GlobalMaxPooling1D(
            $backend,
            [
                'input_shape'=>[4,3]
            ]);

        $layer->build();
        $params = $layer->getParams();
        $this->assertCount(0,$params);

        $grads = $layer->getGrads();
        $this->assertCount(0,$grads);

        $this->assertEquals([3],$layer->outputShape());
    }

    public function testNotspecifiedInputShape()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $layer = new GlobalMaxPooling1D(
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
        $backend = $this->newBackend($mo);
        $layer = new GlobalMaxPooling1D(
            $backend,
            [
            ]);
        $layer->build($inputShape=[4,3]);

        $this->assertEquals([3],$layer->outputShape());
    }

    public function testNormalForwardAndBackward()
    {
        $mo = new MatrixOperator();
        $K = $backend = $this->newBackend($mo);
        $fn = $backend;

        $layer = new GlobalMaxPooling1D(
            $backend,
            ['input_shape'=>[3,2]]);

        $layer->build();

        //
        // forward
        //
        //  batch size 2
        $inputs = $K->array([
            [[1,2],[3,4],[5,6]],
            [[6,5],[4,3],[2,1]],
        ]);
        $copyInputs = $K->copy($inputs);
        $outputs = $layer->forward($inputs, $training=true);
        //
        $this->assertEquals([2,3,2],$inputs->shape());
        $this->assertEquals([2,2],$outputs->shape());
        $this->assertEquals($copyInputs->toArray(),$inputs->toArray());
        $this->assertEquals([
            [5,6],
            [6,5],
        ],$outputs->toArray());

        //
        // backward
        //
        // 2 batch
        $dOutputs = $K->array([
            [1,2],
            [3,4],
        ]);

        $copydOutputs = $K->copy(
            $dOutputs);
        $dInputs = $layer->backward($dOutputs);
        // 2 batch
        $this->assertEquals([2,2],$dOutputs->shape());
        $this->assertEquals([2,3,2],$dInputs->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());
        $this->assertEquals([
            [[0,0],[0,0],[1,2]],
            [[3,4],[0,0],[0,0]],
        ],$dInputs->toArray());
    }
}
