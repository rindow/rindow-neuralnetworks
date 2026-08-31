<?php
namespace RindowTest\NeuralNetworks\Gradient\Func\GatherTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

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

    public function newBackend($nn)
    {
        return $nn->backend();
    }

    public function test1DIndexNormal()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();


        $params = $g->Variable($K->array([
            [1,2,3],
            [4,3,2],
        ]));
        $indices = $g->Variable($K->array([
            2,
            0,
        ],dtype:NDArray::int32));
        $copyParams = $K->copy($params);
        $copyIndices = $K->copy($indices);
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$params,$indices) {
                $outputsVariable = $g->gather($params,$indices,batchDims:1);
                return $outputsVariable;
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        //
        $this->assertEquals([2,3],$params->shape());
        $this->assertEquals([2],$indices->shape());
        $this->assertEquals([2],$outputs->shape());
        $this->assertEquals($copyParams->toArray(),$params->toArray());
        $this->assertEquals($copyIndices->toArray(),$indices->toArray());
        $this->assertEquals([
            3,
            4,
        ],$outputs->toArray());

        $dParams = $tape->gradient($outputsVariable,$params);

        $this->assertEquals([
            [0,0,1],
            [1,0,0],
        ],$dParams->toArray());
    }

    public function test2DIndexNormal()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $params = $g->Variable($K->array([
            [[1,2],[3,4],[5,6]],
            [[6,5],[4,3],[2,1]],
        ]));
        $indices = $g->Variable($K->array([
            [2,2],
            [0,0],
        ],dtype:NDArray::int32));
        $salt = $g->Variable($K->array([
            [1,2],
            [3,4],
        ]));
        $copyParams = $K->copy($params);
        $copyIndices = $K->copy($indices);
        [$outputsVariable,$loss] = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$params,$indices,$salt) {
                $outputsVariable = $g->gather($params,$indices,batchDims:1,detailDepth:3,indexDepth:1);
                return [$outputsVariable,$g->mul($outputsVariable,$salt)];
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        //
        $this->assertEquals([2,3,2],$params->shape());
        $this->assertEquals([2,2],$indices->shape());
        $this->assertEquals([2,2],$outputs->shape());
        $this->assertEquals($copyParams->toArray(),$params->toArray());
        $this->assertEquals($copyIndices->toArray(),$indices->toArray());
        $this->assertEquals([
            [5,6],
            [6,5],
        ],$outputs->toArray());

        $dParams = $tape->gradient($loss,$params);

        $this->assertEquals([
            [[0,0],[0,0],[1,2]],
            [[3,4],[0,0],[0,0]],
        ],$dParams->toArray());
    }

    public function test2DIndexBatchDimsMinusOneNormal()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $params = $g->Variable($K->array([
            [[1,2],[3,4],[5,6]],
            [[6,5],[4,3],[2,1]],
        ]));
        $indices = $g->Variable($K->array([
            [1,1,1],
            [0,0,0],
        ],dtype:NDArray::int32));
        $salt = $g->Variable($K->array([
            [1,2,3],
            [4,5,6],
        ]));
        $copyParams = $K->copy($params);
        $copyIndices = $K->copy($indices);
        [$outputsVariable,$loss] = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$params,$indices,$salt) {
                $outputsVariable = $g->gather($params,$indices,batchDims:-1);
                return [$outputsVariable,$g->mul($outputsVariable,$salt)];
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        //
        $this->assertEquals([2,3,2],$params->shape());
        $this->assertEquals([2,3],$indices->shape());
        $this->assertEquals([2,3],$outputs->shape());
        $this->assertEquals($copyParams->toArray(),$params->toArray());
        $this->assertEquals($copyIndices->toArray(),$indices->toArray());
        $this->assertEquals([
            [2,4,6],
            [6,4,2],
        ],$outputs->toArray());

        $dParams = $tape->gradient($loss,$params);

        $this->assertEquals([
            [[0,1],[0,2],[0,3]],
            [[4,0],[5,0],[6,0]],
        ],$dParams->toArray());
    }

}
