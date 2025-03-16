<?php
namespace RindowTest\NeuralNetworks\Layer\RepeatVectorTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Layer\RepeatVector;
use InvalidArgumentException;

class RepeatVectorTest extends TestCase
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
        $layer = new RepeatVector(
            $K,
            $repeats=2,
            input_shape:[3]
            );

        $inputs = $g->Variable($K->zeros([1,3]));
        $layer->build($inputs);
        $params = $layer->getParams();
        $this->assertCount(0,$params);

        $grads = $layer->getGrads();
        $this->assertCount(0,$grads);

        $this->assertEquals([2,3],$layer->outputShape());
    }

    public function testUnmatchSpecifiedInputShape()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new RepeatVector(
            $K,
            $repeats=2,
            input_shape:[3]
            );

        $inputs = $g->Variable($K->zeros([1,5]));
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Input shape is inconsistent: defined as (3) but (5) given in RepeatVector');
        $layer->build($inputs);
    }

    public function testInvalidInputShape()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new RepeatVector(
            $K,
            $repeats=2,
            input_shape:[3,2]
            );

        $inputs = $g->Variable($K->zeros([1,3,2]));
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('input shape must be 1D');
        $layer->build($inputs);
    }

    public function testNormalForwardAndBackward()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $fn = $K;

        $layer = new RepeatVector(
            $K,
            $repeats=2,
            input_shape:[3]);

        //$layer->build();

        //
        // forward
        //
        //  batch size 2
        $inputs = $K->array($mo->arange(2*3,null,null,NDArray::float32)->reshape([2,3]));
        $copyInputs = $K->copy($inputs);
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$inputs) {
                $outputsVariable = $layer->forward($inputs);
                return $outputsVariable;
            }
        );
        $outputs = $K->ndarray($outputsVariable);
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
        [$dInputs] = $outputsVariable->creator()->backward([$dOutputs]);
        // 2 batch
        $this->assertEquals([2,3],$dInputs->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());
        $this->assertEquals([
            [0,2,4],
            [6,8,10],
        ],$dInputs->toArray());
    }
}
