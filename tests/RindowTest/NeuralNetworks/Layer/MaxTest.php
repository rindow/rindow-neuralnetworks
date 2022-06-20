<?php
namespace RindowTest\NeuralNetworks\Layer\MaxTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Layer\Max;
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
        $layer = new Max(
            $K,
            input_shape:[3]
            );

        $inputs = $g->Variable($K->zeros([1,3]));
        $layer->build($inputs);
        $params = $layer->getParams();
        $this->assertCount(0,$params);

        $grads = $layer->getGrads();
        $this->assertCount(0,$grads);

        $this->assertEquals([],$layer->outputShape());
    }

    public function testSetInputShape()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new Max(
            $K,
            );
        $inputs = $g->Variable($K->zeros([1,3]));
        $layer->build($inputs);

        $this->assertEquals([],$layer->outputShape());
    }

    public function testUnmatchSpecifiedInputShape()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new Max(
            $K,
            input_shape:[3]
            );

        $inputs = $g->Variable($K->zeros([1,5]));
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Input shape is inconsistent: defined as [3] but [5] given in Max');
        $layer->build($inputs);
    }

    public function test1DNormalForwardAndBackward()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $fn = $K;

        $layer = new Max(
            $K,
            input_shape:[3]);

        //$layer->build();

        //
        // forward
        //
        //  batch size 2
        $inputs = $K->array([
            [1,2,3],
            [4,3,2],
        ]);
        $copyInputs = $K->copy($inputs);
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$inputs) {
                $outputsVariable = $layer->forward($inputs, $training=true);
                return $outputsVariable;
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        //
        $this->assertEquals([2,3],$inputs->shape());
        $this->assertEquals([2],$outputs->shape());
        $this->assertEquals($copyInputs->toArray(),$inputs->toArray());
        $this->assertEquals([
            3,
            4,
        ],$outputs->toArray());

        //
        // backward
        //
        // 2 batch
        $dOutputs = $K->array([
            2,
            3,
        ]);

        $copydOutputs = $K->copy(
            $dOutputs);
        [$dInputs] = $outputsVariable->creator()->backward([$dOutputs]);
        // 2 batch
        $this->assertEquals([2],$dOutputs->shape());
        $this->assertEquals([2,3],$dInputs->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());
        $this->assertEquals([
            [0,0,2],
            [3,0,0],
        ],$dInputs->toArray());
    }

    public function test2DNormalForwardAndBackward()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $fn = $K;

        $layer = new Max(
            $K,
            axis:0,input_shape:[3,2]);

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
                $outputsVariable = $layer->forward($inputs, $training=true);
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
