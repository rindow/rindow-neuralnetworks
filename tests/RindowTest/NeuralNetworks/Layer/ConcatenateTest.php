<?php
namespace RindowTest\NeuralNetworks\Layer\ConcatenateTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Layer\Concatenate;
use InvalidArgumentException;

class ConcatenateTest extends TestCase
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
        $layer = new Concatenate($K,
                #axis:-1,
                input_shapes:[[4,3],[4,2]],
        );
        $inputs = [
            $g->Variable($K->zeros([1,4,3])),
            $g->Variable($K->zeros([1,4,2])),
        ];

        $layer->build($inputs);
        $params = $layer->getParams();
        $this->assertCount(0,$params);

        $grads = $layer->getGrads();
        $this->assertCount(0,$grads);

        $this->assertEquals([4,5],$layer->outputShape());
    }

    public function testUnmatchSpecifiedInputShape()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new Concatenate($K,input_shapes:[[4,3],[4,2]]);
        $inputs = [
            $g->Variable($K->zeros([1,4,5])),
            $g->Variable($K->zeros([1,4,2])),
        ];
    
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Input shape is inconsistent: defined as [[4,3],[4,2]] but [[4,5],[4,2]] given in Concatenate');
        $layer->build($inputs);
    }

    public function testSetInputShape()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new Concatenate($K,axis:1);
        // [batch,2,4],[batch,3,4]
        $inputs = [
            $g->Variable($K->zeros([1,2,4])),
            $g->Variable($K->zeros([1,3,4])),
        ];
        $layer->build($inputs);
        // [batch,5,4]
        $this->assertEquals([5,4],$layer->outputShape());
    }

    public function testNormalForwardAndBackward()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();

        $layer = new Concatenate($K,
                #axis:-1,
        );

        //  batch size 2
        $i1 = $K->array($mo->arange(2*2*2,null,null,NDArray::float32)->reshape([2,2,2]));
        $i2 = $K->array($mo->arange(2*2*3,100,null,NDArray::float32)->reshape([2,2,3]));
        $inputs = [$i1,$i2];

        $layer->build([$g->Variable($i1),$g->Variable($i2)]);

        //
        // forward
        //
        $copyInputs = [$K->copy($i1),$K->copy($i2)];
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$inputs) {
                $outputsVariable = $layer->forward($inputs);
                return $outputsVariable;
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        //
        $this->assertEquals([2,2,5],$outputs->shape());
        $this->assertEquals($copyInputs[0]->toArray(),$inputs[0]->toArray());
        $this->assertEquals($copyInputs[1]->toArray(),$inputs[1]->toArray());
        $this->assertEquals([
            [[0,1,100,101,102],[2,3,103,104,105]],
            [[4,5,106,107,108],[6,7,109,110,111]],
        ],$outputs->toArray());
        //
        // backward
        //
        // 2 batch
        $dOutputs = $K->copy($outputsVariable->value());

        $copydOutputs = $K->copy(
            $dOutputs);
        $dInputs = $outputsVariable->creator()->backward([$dOutputs]);
        // 2 batch
        $this->assertCount(2,$dInputs);
        $this->assertEquals([2,2,2],$dInputs[0]->shape());
        $this->assertEquals([2,2,3],$dInputs[1]->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());
        $this->assertEquals([
            [[0,1],[2,3]],
            [[4,5],[6,7]],
        ],$dInputs[0]->toArray());
        $this->assertEquals([
            [[100,101,102],[103,104,105]],
            [[106,107,108],[109,110,111]],
        ],$dInputs[1]->toArray());
    }
}
