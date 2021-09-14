<?php
namespace RindowTest\NeuralNetworks\Layer\EmbeddingTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Layer\Embedding;
use Rindow\NeuralNetworks\Gradient\Core\Undetermined;
use Rindow\NeuralNetworks\Gradient\Core\UndeterminedNDArray;
use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;

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
        $backend = $this->newBackend($mo);
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
        $backend = $this->newBackend($mo);
        $layer = new Embedding(
            $backend,
            $inputDim=4,
            $outputDim=5,
            [
            ]);
        $layer->build($this->newInputShape([3]));

        //$this->assertEquals([3],$layer->inputShape());
        $this->assertEquals([3,5],$layer->outputShape());
    }

    public function testNormalForwardAndBackward()
    {
        $mo = new MatrixOperator();
        $K = $backend = $this->newBackend($mo);
        $fn = $backend;

        $layer = new Embedding(
            $backend,
            $inputDim=4,
            $outputDim=5,
            ['input_length'=>3]);

        $kernel = $K->array($mo->arange(4*5,null,null,NDArray::float32)->reshape([4,5]));
        $layer->build(null,
            ['sampleWeights'=>[$kernel]]
        );


        //
        // forward
        //
        //  2 batch
        $inputs = $K->array([
            [0,1,2],
            [3,2,1],
        ]);
        $copyInputs = $K->copy($inputs);
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
            $K->ones([2,3,5]);

        $copydOutputs = $K->copy(
            $dOutputs);
        [$dInputs] = $layer->backward([$dOutputs]);
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
