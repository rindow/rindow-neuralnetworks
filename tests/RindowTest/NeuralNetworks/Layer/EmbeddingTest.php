<?php
namespace RindowTest\NeuralNetworks\Layer\EmbeddingTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Layer\Embedding;
use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;

class Test extends TestCase
{
    public function testDefaultInitialize()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $layer = new Embedding(
            $backend,
            $inputDim=4,
            $outputDim=5,
            [
                'input_length'=>3
            ]);

        $layer->build();
        $params = $layer->getParams();
        $this->assertCount(1,$params);
        $this->assertEquals([4,5],$params[0]->shape());

        $grads = $layer->getGrads();
        $this->assertCount(1,$grads);
        $this->assertEquals([4,5],$grads[0]->shape());

        //$this->assertEquals([3],$layer->inputShape());
        $this->assertEquals([3,5],$layer->outputShape());
    }

    public function testNotspecifiedInputShape()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $layer = new Embedding(
            $backend,
            $inputDim=4,
            $outputDim=5,
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
        $layer = new Embedding(
            $backend,
            $inputDim=4,
            $outputDim=5,
            [
            ]);
        $layer->build($inputShape=[3]);

        //$this->assertEquals([3],$layer->inputShape());
        $this->assertEquals([3,5],$layer->outputShape());
    }

    public function testNormalForwardAndBackward()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $fn = $backend;

        $layer = new Embedding(
            $backend,
            $inputDim=4,
            $outputDim=5,
            ['input_length'=>3]);

        $kernel = $mo->arange(4*5,null,null,NDArray::float32)->reshape([4,5]);
        $layer->build(null,
            ['sampleWeights'=>[$kernel]]
        );


        //
        // forward
        //
        //  2 batch
        $inputs = $mo->array([
            [0,1,2],
            [3,2,1],
        ]);
        $copyInputs = $mo->copy($inputs);
        $outputs = $layer->forward($inputs, $training=true);
        //
        $this->assertEquals([
            [[0,1,2,3,4],
             [5,6,7,8,9],
             [10,11,12,13,14]],
            [[15,16,17,18,19],
             [10,11,12,13,14],
             [5,6,7,8,9]],
        ],$outputs->toArray());
        $this->assertEquals($copyInputs->toArray(),$inputs->toArray());

        //
        // backward
        //
        // 2 batch
        $dOutputs =
            $mo->ones([2,3,5]);

        $copydOutputs = $mo->copy(
            $dOutputs);
        $dInputs = $layer->backward($dOutputs);
        // 2 batch
        $this->assertEquals([2,3],$dInputs->shape());
        $grads = $layer->getGrads();
        $this->assertEquals([
            [1,1,1,1,1],
            [2,2,2,2,2],
            [2,2,2,2,2],
            [1,1,1,1,1],
        ],$grads[0]->toArray());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());
    }
}
