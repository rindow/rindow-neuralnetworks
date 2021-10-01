<?php
namespace RindowTest\NeuralNetworks\Layer\ExpandDimsTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Layer\ExpandDims;
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

    public function testDefaultInitializePlus()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $layer = new ExpandDims(
            $backend,
            $axis=1,
            [
                'input_shape'=>[4,4,3]
            ]);

        $layer->build();
        $params = $layer->getParams();
        $this->assertCount(0,$params);

        $grads = $layer->getGrads();
        $this->assertCount(0,$grads);

        $this->assertEquals([4,1,4,3],$layer->outputShape());
    }

    public function testDefaultInitializePlus3()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $layer = new ExpandDims(
            $backend,
            $axis=3,
            [
                'input_shape'=>[4,4,3]
            ]);

        $layer->build();
        $params = $layer->getParams();
        $this->assertCount(0,$params);

        $grads = $layer->getGrads();
        $this->assertCount(0,$grads);

        $this->assertEquals([4,4,3,1],$layer->outputShape());
    }

    public function testDefaultInitializeMinus()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $layer = new ExpandDims(
            $backend,
            $axis=-1,
            [
                'input_shape'=>[4,4,3]
            ]);

        $layer->build();
        $params = $layer->getParams();
        $this->assertCount(0,$params);

        $grads = $layer->getGrads();
        $this->assertCount(0,$grads);

        $this->assertEquals([4,4,3,1],$layer->outputShape());
    }

    public function testDefaultInitializeMinus2()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $layer = new ExpandDims(
            $backend,
            $axis=-2,
            [
                'input_shape'=>[4,4,3]
            ]);

        $layer->build();
        $params = $layer->getParams();
        $this->assertCount(0,$params);

        $grads = $layer->getGrads();
        $this->assertCount(0,$grads);

        $this->assertEquals([4,4,1,3],$layer->outputShape());
    }

    public function testDefaultInitializePlusOver()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $layer = new ExpandDims(
            $backend,
            $axis=4,
            [
                'input_shape'=>[4,4,3]
            ]);

        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Invalid axis. Dims of the inputshape is 3. axis=4 given');
        $layer->build();
    }

    public function testDefaultInitializeMinusOver()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $layer = new ExpandDims(
            $backend,
            $axis=-5,
            [
                'input_shape'=>[4,4,3]
            ]);

        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Invalid axis. Dims of the inputshape is 3. axis=-5 given');
        $layer->build();
    }

    public function testNotspecifiedInputShape()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $layer = new ExpandDims(
            $backend,
            $axis=1,
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
        $layer = new ExpandDims(
            $backend,
            $axis=0,
            [
            ]);
        $layer->build($this->newInputShape([4,4,3]));

        $this->assertEquals([1,4,4,3],$layer->outputShape());
    }

    public function testNormalForwardAndBackward()
    {
        $mo = new MatrixOperator();
        $K = $backend = $this->newBackend($mo);
        $fn = $backend;

        $layer = new ExpandDims(
            $backend,
            $axis=0,
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
            [2,1,4,4,3],$outputs->shape());
        $this->assertEquals($copyInputs->toArray(),$inputs->toArray());

        //
        // backward
        //
        // 2 batch
        $dOutputs =
            $K->array($mo->arange(2*4*4*3)->reshape([2,1,4,4,3]));

        $copydOutputs = $K->copy(
            $dOutputs);
        [$dInputs] = $layer->backward([$dOutputs]);
        // 2 batch
        $this->assertEquals([2,4,4,3],$dInputs->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());
    }
}
