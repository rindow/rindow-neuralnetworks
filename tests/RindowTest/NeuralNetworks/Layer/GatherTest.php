<?php
namespace RindowTest\NeuralNetworks\Layer\GatherTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Layer\Gather;
use Rindow\NeuralNetworks\Gradient\Core\Undetermined;
use Rindow\NeuralNetworks\Gradient\Core\UndeterminedNDArray;
use InvalidArgumentException;

class Test extends TestCase
{
    public function newBackend($mo)
    {
        $builder = new NeuralNetworks($mo);
        return $builder->backend();
    }

    public function newInputShape($inputShape)
    {
        array_unshift($inputShape,1);
        $variable = new Undetermined(new UndeterminedNDArray($inputShape));
        return $variable;
    }

    public function testDefaultInitialize()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $layer = new Gather(
            $backend,
            [
                'input_shapes'=>[[3],[]]
            ]);

        $layer->build();
        $params = $layer->getParams();
        $this->assertCount(0,$params);

        $grads = $layer->getGrads();
        $this->assertCount(0,$grads);

        $this->assertEquals([],$layer->outputShape());
    }

    public function test2DInitialize()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $layer = new Gather(
            $backend,
            [
                'axis'=>0,
                'input_shapes'=>[[3,2],[2]]
            ]);

        $layer->build();
        $params = $layer->getParams();
        $this->assertCount(0,$params);

        $grads = $layer->getGrads();
        $this->assertCount(0,$grads);

        $this->assertEquals([2],$layer->outputShape());
    }

    public function testNotspecifiedInputShape()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $layer = new Gather(
            $backend,
            [
            ]);

        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Input shape is not defined');
        $layer->build();
    }

    public function testNullAxis()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $layer = new Gather(
            $backend,
            [
                'axis'=>null,
                'input_shapes'=>[[3,2],[2]]
            ]);

        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Null axis is not supported.');
        $layer->build();
    }

    public function testSetInputShape()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $layer = new Gather(
            $backend,
            [
            ]);
        $layer->build([$this->newInputShape([3]),$this->newInputShape([])]);

        $this->assertEquals([],$layer->outputShape());
    }

    public function test1DNormalForwardAndBackward()
    {
        $mo = new MatrixOperator();
        $K = $backend = $this->newBackend($mo);
        $fn = $backend;

        $layer = new Gather(
            $backend,
            ['input_shapes'=>[[3],[]]]);

        $layer->build();

        //
        // forward
        //
        //  batch size 2
        $sources = $K->array([
            [1,2,3],
            [4,3,2],
        ]);
        $indexes = $K->array([
            2,
            0,
        ]);
        $copySources = $K->copy($sources);
        $copyIndexes = $K->copy($indexes);
        $outputs = $layer->forward([$sources,$indexes], $training=true);
        //
        $this->assertEquals([2,3],$sources->shape());
        $this->assertEquals([2],$indexes->shape());
        $this->assertEquals([2],$outputs->shape());
        $this->assertEquals($copySources->toArray(),$sources->toArray());
        $this->assertEquals($copyIndexes->toArray(),$indexes->toArray());
        $this->assertEquals([
            3,
            4,
        ],$outputs->toArray());

        //
        // backward
        //
        // 2 batch
        $dOutputs = $K->array([
            2,
            3,
        ]);

        $copydOutputs = $K->copy(
            $dOutputs);
        [$dSources,$dIndexes] = $layer->backward([$dOutputs]);
        // 2 batch
        $this->assertEquals([2],$dOutputs->shape());
        $this->assertEquals([2,3],$dSources->shape());
        $this->assertEquals([2],$dIndexes->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());
        $this->assertEquals([
            [0,0,2],
            [3,0,0],
        ],$dSources->toArray());
        $this->assertEquals([
            0,
            0,
        ],$dIndexes->toArray());
    }

    public function test2DNormalForwardAndBackward()
    {
        $mo = new MatrixOperator();
        $K = $backend = $this->newBackend($mo);
        $fn = $backend;

        $layer = new Gather(
            $backend,
            ['axis'=>0,'input_shapes'=>[[3,2],[2]]]);

        $layer->build();

        //
        // forward
        //
        //  batch size 2
        $sources = $K->array([
            [[1,2],[3,4],[5,6]],
            [[6,5],[4,3],[2,1]],
        ]);
        $indexes = $K->array([
            [2,2],
            [0,0],
        ]);
        $copySources = $K->copy($sources);
        $copyIndexes = $K->copy($indexes);
        $outputs = $layer->forward([$sources,$indexes], $training=true);
        //
        $this->assertEquals([2,3,2],$sources->shape());
        $this->assertEquals([2,2],$indexes->shape());
        $this->assertEquals([2,2],$outputs->shape());
        $this->assertEquals($copySources->toArray(),$sources->toArray());
        $this->assertEquals($copyIndexes->toArray(),$indexes->toArray());
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
        [$dSources,$dIndexes] = $layer->backward([$dOutputs]);
        // 2 batch
        $this->assertEquals([2,2],$dOutputs->shape());
        $this->assertEquals([2,3,2],$dSources->shape());
        $this->assertEquals([2,2],$dIndexes->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());
        $this->assertEquals([
            [[0,0],[0,0],[1,2]],
            [[3,4],[0,0],[0,0]],
        ],$dSources->toArray());
        $indexes = $K->array([
            [0,0],
            [0,0],
        ]);
    }
}
