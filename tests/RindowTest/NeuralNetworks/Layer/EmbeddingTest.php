<?php
namespace RindowTest\NeuralNetworks\Layer\EmbeddingTest;

use InvalidArgumentException;
use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Layer\Embedding;
use Interop\Polite\Math\Matrix\NDArray;

class EmbeddingTest extends TestCase
{
    public function newMatrixOperator()
    {
        return new MatrixOperator();
    }

    public function newNeuralNetworks($mo)
    {
        return new NeuralNetworks($mo);
    }

    public function testDefaultInitialize()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new Embedding(
            $K,
            $inputDim=4,
            $outputDim=5,
            input_length:3
            );

        $inputs = $g->Variable($K->zeros([1,3]));
        $layer->build($inputs);
        $params = $layer->getParams();
        $this->assertCount(1,$params);
        $this->assertEquals([4,5],$params[0]->shape());

        $grads = $layer->getGrads();
        $this->assertCount(1,$grads);
        $this->assertEquals([4,5],$grads[0]->shape());

        //$this->assertEquals([3],$layer->inputShape());
        $this->assertEquals([3,5],$layer->outputShape());
    }

    public function testSetInputShape()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new Embedding(
            $K,
            $inputDim=4,
            $outputDim=5,
            );
        $inputs = $g->Variable($K->zeros([1,3]));
        $layer->build($inputs);

        //$this->assertEquals([3],$layer->inputShape());
        $this->assertEquals([3,5],$layer->outputShape());
    }

    public function testUnmatchSpecifiedInputShape()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new Embedding(
            $K,
            $inputDim=4,
            $outputDim=5,
            input_length:3
            );
        $inputs = $g->Variable($K->zeros([1,4]));

        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Input shape is inconsistent: defined as [3] but [4] given in Embedding');
        $layer->build($inputs);
    }

    public function testNormalForwardAndBackward()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();

        $layer = new Embedding(
            $K,
            $inputDim=4,
            $outputDim=5,
            input_length:3);
        //  2 batch
        $inputs = $K->array([
            [0,1,2],
            [3,2,1],
        ]);

        $kernel = $K->array($mo->arange(4*5,null,null,NDArray::float32)->reshape([4,5]));
        $layer->build($g->Variable($inputs),
            sampleWeights:[$kernel]
        );


        //
        // forward
        //
        $copyInputs = $K->copy($inputs);
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$inputs) {
                $outputsVariable = $layer->forward($inputs);
                return $outputsVariable;
            }
        );
        $outputs = $K->ndarray($outputsVariable);
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
        [$dInputs] = $outputsVariable->creator()->backward([$dOutputs]);
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
