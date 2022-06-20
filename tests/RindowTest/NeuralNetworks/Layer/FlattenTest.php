<?php
namespace RindowTest\NeuralNetworks\Layer\FlattenTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Layer\Flatten;
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
        $layer = new Flatten(
            $backend,
            [
                'input_shape'=>[4,4,3]
            ]);

        $layer->build();
        $params = $layer->getParams();
        $this->assertCount(0,$params);

        $grads = $layer->getGrads();
        $this->assertCount(0,$grads);

        $this->assertEquals([48],$layer->outputShape());
    }

    public function testNotspecifiedInputShape()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $layer = new Flatten(
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
        $layer = new Flatten(
            $backend,
            [
            ]);
        $layer->build($this->newInputShape([4,4,3]));

        $this->assertEquals([48],$layer->outputShape());
    }

    public function testNormalForwardAndBackward()
    {
        $mo = new MatrixOperator();
        $K = $backend = $this->newBackend($mo);
        $fn = $backend;

        $layer = new Flatten(
            $backend,
            ['input_shape'=>[4,4,3]]);

        $layer->build();

        //
        // forward
        //
        //  batch size 2
        $inputs = $K->array($mo->arange(2*4*4*3)->reshape([2,4,4,3]));
        $copyInputs = $K->copy($inputs);
        $outputs = $layer->forward($inputs, $training=true);
        //
        $this->assertEquals(
            [2,48],$outputs->shape());
        $this->assertEquals($copyInputs->toArray(),$inputs->toArray());

        //
        // backward
        //
        // 2 batch
        $dOutputs =
            $K->array($mo->arange(2*4*4*3)->reshape([2,4*4*3]));

        $copydOutputs = $K->copy(
            $dOutputs);
        [$dInputs] = $layer->backward([$dOutputs]);
        // 2 batch
        $this->assertEquals([2,4,4,3],$dInputs->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());
    }
}
