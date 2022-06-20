<?php
namespace RindowTest\NeuralNetworks\Layer\GlobalAveragePooling1DTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Layer\GlobalAveragePooling1D;
use InvalidArgumentException;

class Test extends TestCase
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
        $layer = new GlobalAveragePooling1D(
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
        $layer = new GlobalAveragePooling1D(
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
        $layer = new GlobalAveragePooling1D(
            $K,
            input_shape:[4,3]
            );

        $inputs = $g->Variable($K->zeros([1,4,5]));
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Input shape is inconsistent: defined as [4,3] but [4,5] given in GlobalAveragePooling1D');
        $layer->build($inputs);
    }

    public function testNormalForwardAndBackward()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $fn = $K;

        $layer = new GlobalAveragePooling1D(
            $K,
            input_shape:[2,3]);

        //$layer->build();

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
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$inputs) {
                $outputsVariable = $layer->forward($inputs, $training=true);
                return $outputsVariable;
            }
        );
        $outputs = $K->ndarray($outputsVariable);
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
        [$dInputs] = $outputsVariable->creator()->backward([$dOutputs]);
        // 2 batch
        $this->assertEquals([2,2,3],$dInputs->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());
        $this->assertEquals([
            [[0.5,1,1.5],[0.5,1,1.5]],
            [[1,1.5,2],[1,1.5,2]],
        ],$dInputs->toArray());
    }
}
