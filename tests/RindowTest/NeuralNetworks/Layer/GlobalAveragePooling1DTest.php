<?php
namespace RindowTest\NeuralNetworks\Layer\GlobalAveragePooling1DTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Layer\GlobalAveragePooling1D;
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
        $layer = new GlobalAveragePooling1D(
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
        $layer = new GlobalAveragePooling1D(
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
        $layer = new GlobalAveragePooling1D(
            $backend,
            [
            ]);
        $layer->build($this->newInputShape([4,3]));

        $this->assertEquals([3],$layer->outputShape());
    }

    public function testNormalForwardAndBackward()
    {
        $mo = new MatrixOperator();
        $K = $backend = $this->newBackend($mo);
        $fn = $backend;

        $layer = new GlobalAveragePooling1D(
            $backend,
            ['input_shape'=>[2,3]]);

        $layer->build();

        //
        // forward
        //
        //  batch size 2
        $inputs = $K->array([
            [[1,2,3],[2,3,4]],
            [[2,3,4],[3,4,5]],
        ]);
        $this->assertEquals([2,2,3],$inputs->shape());
        $copyInputs = $K->copy($inputs);
        $outputs = $layer->forward($inputs, $training=true);
        //
        $this->assertEquals([2,3],$outputs->shape());
        $this->assertEquals($copyInputs->toArray(),$inputs->toArray());
        $this->assertEquals([
            [1.5,2.5,3.5],
            [2.5,3.5,4.5],
        ],$outputs->toArray());

        //
        // backward
        //
        // 2 batch
        $dOutputs = $K->array([
            [1,2,3],
            [2,3,4],
        ]);
        $copydOutputs = $K->copy($dOutputs);
        [$dInputs] = $layer->backward([$dOutputs]);
        // 2 batch
        $this->assertEquals([2,2,3],$dInputs->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());
        $this->assertEquals([
            [[0.5,1,1.5],[0.5,1,1.5]],
            [[1,1.5,2],[1,1.5,2]],
        ],$dInputs->toArray());
    }
}
