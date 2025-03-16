<?php
namespace RindowTest\NeuralNetworks\Layer\GlobalMaxPooling1DTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Layer\GlobalMaxPooling1D;
use InvalidArgumentException;

class GlobalMaxPooling1DTest extends TestCase
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
        $layer = new GlobalMaxPooling1D(
            $K,
            input_shape:[4,3]
            );

        $inputs = $g->Variable($K->zeros([1,4,3]));
        $layer->build($inputs);
        $params = $layer->getParams();
        $this->assertCount(0,$params);

        $grads = $layer->getGrads();
        $this->assertCount(0,$grads);

        $this->assertEquals([3],$layer->outputShape());
    }

    public function testSetInputShape()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new GlobalMaxPooling1D(
            $K,
            );
        $inputs = $g->Variable($K->zeros([1,4,3]));
        $layer->build($inputs);

        $this->assertEquals([3],$layer->outputShape());
    }

    public function testUnmatchSpecifiedInputShape()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new GlobalMaxPooling1D(
            $K,
            input_shape:[4,3]
            );

        $inputs = $g->Variable($K->zeros([1,4,5]));
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Input shape is inconsistent: defined as (4,3) but (4,5) given in GlobalMaxPooling1D');
        $layer->build($inputs);
    }

    public function testNormalForwardAndBackward()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $fn = $K;

        $layer = new GlobalMaxPooling1D(
            $K,
            input_shape:[3,2]);

        //$layer->build();

        //
        // forward
        //
        //  batch size 2
        $inputs = $K->array([
            [[1,2],[3,4],[5,6]],
            [[6,5],[4,3],[2,1]],
        ]);
        $copyInputs = $K->copy($inputs);
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$inputs) {
                $outputsVariable = $layer->forward($inputs);
                return $outputsVariable;
            }
        );
        $outputs = $K->ndarray($outputsVariable);
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
        [$dInputs] = $outputsVariable->creator()->backward([$dOutputs]);
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
