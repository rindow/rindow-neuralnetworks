<?php
namespace RindowTest\NeuralNetworks\Layer\GatherTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Layer\Gather;
use InvalidArgumentException;

class GatherTest extends TestCase
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
        $layer = new Gather(
            $K,
            input_shapes:[[3],[]]
            );

        $inputs = [$g->Variable($K->zeros([1,3])),$g->Variable($K->zeros([1]))];
        $layer->build($inputs);
        $params = $layer->getParams();
        $this->assertCount(0,$params);

        $grads = $layer->getGrads();
        $this->assertCount(0,$grads);

        $this->assertEquals([],$layer->outputShape());
    }

    public function test2DInitialize()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new Gather(
            $K,
            axis:0,
            input_shapes:[[3,2],[2]]
            );

        $inputs = [$g->Variable($K->zeros([1,3,2])),$g->Variable($K->zeros([1,2]))];
        $layer->build($inputs);
        $params = $layer->getParams();
        $this->assertCount(0,$params);

        $grads = $layer->getGrads();
        $this->assertCount(0,$grads);

        $this->assertEquals([2],$layer->outputShape());
    }

    public function testNullAxis()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new Gather(
            $K,
            axis:null,
            input_shapes:[[3,2],[2]]
            );

        $inputs = [$g->Variable($K->zeros([1,3,2])),$g->Variable($K->zeros([1,2]))];
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Unmatch source and index Shape and axis:[3,2],[3],-1');
        $layer->build($inputs);
    }

    public function testSetInputShape()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new Gather(
            $K,
            );

        $inputs = [$g->Variable($K->zeros([1,3])),$g->Variable($K->zeros([1]))];
        $layer->build($inputs);

        $this->assertEquals([],$layer->outputShape());
    }

    public function testUnmatchSpecifiedInputShape()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new Gather(
            $K,
            input_shapes:[[3,2],[2]],
            );

        $inputs = [$g->Variable($K->zeros([1,3,2])),$g->Variable($K->zeros([1,4]))];
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Input shape is inconsistent: defined as [[3,2],[2]] but [[3,2],[4]] given in Gather');
        $layer->build($inputs);
    }

    public function test1DNormalForwardAndBackward()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $fn = $K;

        $layer = new Gather(
            $K,
            input_shapes:[[3],[]]);

        //$layer->build($g->Variable($inputs));

        //
        // forward
        //
        //  batch size 2
        $sources = $K->array([
            [1,2,3],
            [4,3,2],
        ]);
        $indexes = $K->array([
            2,
            0,
        ]);
        $copySources = $K->copy($sources);
        $copyIndexes = $K->copy($indexes);
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$sources,$indexes) {
                $outputsVariable = $layer->forward([$sources,$indexes]);
                return $outputsVariable;
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        //
        $this->assertEquals([2,3],$sources->shape());
        $this->assertEquals([2],$indexes->shape());
        $this->assertEquals([2],$outputs->shape());
        $this->assertEquals($copySources->toArray(),$sources->toArray());
        $this->assertEquals($copyIndexes->toArray(),$indexes->toArray());
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
        [$dSources,$dIndexes] = $outputsVariable->creator()->backward([$dOutputs]);
        // 2 batch
        $this->assertEquals([2],$dOutputs->shape());
        $this->assertEquals([2,3],$dSources->shape());
        $this->assertEquals([2],$dIndexes->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());
        $this->assertEquals([
            [0,0,2],
            [3,0,0],
        ],$dSources->toArray());
        $this->assertEquals([
            0,
            0,
        ],$dIndexes->toArray());
    }

    public function test2DNormalForwardAndBackward()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $fn = $K;

        $layer = new Gather(
            $K,
            axis:0,input_shapes:[[3,2],[2]]);

        //$layer->build();

        //
        // forward
        //
        //  batch size 2
        $sources = $K->array([
            [[1,2],[3,4],[5,6]],
            [[6,5],[4,3],[2,1]],
        ]);
        $indexes = $K->array([
            [2,2],
            [0,0],
        ]);
        $copySources = $K->copy($sources);
        $copyIndexes = $K->copy($indexes);
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$sources,$indexes) {
                $outputsVariable = $layer->forward([$sources,$indexes]);
                return $outputsVariable;
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        //
        $this->assertEquals([2,3,2],$sources->shape());
        $this->assertEquals([2,2],$indexes->shape());
        $this->assertEquals([2,2],$outputs->shape());
        $this->assertEquals($copySources->toArray(),$sources->toArray());
        $this->assertEquals($copyIndexes->toArray(),$indexes->toArray());
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
        [$dSources,$dIndexes] = $outputsVariable->creator()->backward([$dOutputs]);
        // 2 batch
        $this->assertEquals([2,2],$dOutputs->shape());
        $this->assertEquals([2,3,2],$dSources->shape());
        $this->assertEquals([2,2],$dIndexes->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());
        $this->assertEquals([
            [[0,0],[0,0],[1,2]],
            [[3,4],[0,0],[0,0]],
        ],$dSources->toArray());
        $indexes = $K->array([
            [0,0],
            [0,0],
        ]);
    }
}
