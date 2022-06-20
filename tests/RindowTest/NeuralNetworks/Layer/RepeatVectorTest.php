<?php
namespace RindowTest\NeuralNetworks\Layer\RepeatVectorTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Layer\RepeatVector;
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
        $layer = new RepeatVector(
            $backend,
            $repeats=2,
            [
                'input_shape'=>[3]
            ]);

        $layer->build();
        $params = $layer->getParams();
        $this->assertCount(0,$params);

        $grads = $layer->getGrads();
        $this->assertCount(0,$grads);

        $this->assertEquals([2,3],$layer->outputShape());
    }

    public function testNotspecifiedInputShape()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $layer = new RepeatVector(
            $backend,
            $repeats=2,
            [
            ]);

        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Input shape is not defined');
        $layer->build();
    }

    public function testInvalidInputShape()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $layer = new RepeatVector(
            $backend,
            $repeats=2,
            ['input_shape'=>[3,2]
            ]);

        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('input shape must be 1D');
        $layer->build();
    }

    public function testNormalForwardAndBackward()
    {
        $mo = new MatrixOperator();
        $K = $backend = $this->newBackend($mo);
        $fn = $backend;

        $layer = new RepeatVector(
            $backend,
            $repeats=2,
            ['input_shape'=>[3]]);

        $layer->build();

        //
        // forward
        //
        //  batch size 2
        $inputs = $K->array($mo->arange(2*3,null,null,NDArray::float32)->reshape([2,3]));
        $copyInputs = $K->copy($inputs);
        $outputs = $layer->forward($inputs, $training=true);
        //
        $this->assertEquals([2,2,3],$outputs->shape());
        $this->assertEquals($copyInputs->toArray(),$inputs->toArray());
        $this->assertEquals([
            [[0,1,2],[0,1,2]],
            [[3,4,5],[3,4,5]],
        ],$outputs->toArray());
        //
        // backward
        //
        // 2 batch
        $dOutputs = $K->array([
            [[0,1,2],[0,1,2]],
            [[3,4,5],[3,4,5]],
        ],NDArray::float32)->reshape([2,2,3]);

        $copydOutputs = $K->copy(
            $dOutputs);
        [$dInputs] = $layer->backward([$dOutputs]);
        // 2 batch
        $this->assertEquals([2,3],$dInputs->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());
        $this->assertEquals([
            [0,2,4],
            [6,8,10],
        ],$dInputs->toArray());
    }
}
