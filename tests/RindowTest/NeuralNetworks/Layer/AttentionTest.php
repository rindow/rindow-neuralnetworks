<?php
namespace RindowTest\NeuralNetworks\Layer\AttentionTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Layer\Attention;
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
        $layer = new Attention($K, input_shapes:[[3,2],[4,2]]);
        $inputs = [
            $g->Variable($K->zeros([1,3,2])),
            $g->Variable($K->zeros([1,4,2])),
        ];

        $layer->build($inputs);
        $params = $layer->getParams();
        $this->assertCount(0,$params);

        $grads = $layer->getGrads();
        $this->assertCount(0,$grads);

        $this->assertEquals([3,2],$layer->outputShape());
    }

    public function testInitializeWithReturnAttentionScores()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new Attention($K, input_shapes:[[3,2],[4,2]]);
        $inputs = [
            $g->Variable($K->zeros([1,3,2])),
            $g->Variable($K->zeros([1,4,2])),
        ];

        $shapes = $layer->build($inputs);
        $params = $layer->getParams();
        $this->assertCount(0,$params);

        $grads = $layer->getGrads();
        $this->assertCount(0,$grads);

        $this->assertEquals([3,2],$layer->outputShape());
    }

    public function testSetInputShape()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new Attention($K, input_shapes:[[3,2],[4,2]]);
        $inputs = [
            $g->Variable($K->zeros([1,3,2])),
            $g->Variable($K->zeros([1,4,2])),
        ];
        // [batch,3,2],[batch,4,2]
        $layer->build($inputs);
        // [batch,3,4]
        $this->assertEquals([3,2],$layer->outputShape());
    }

    public function testUnmatchSpecifiedInputShape()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new Attention($K, input_shapes:[[3,2],[4,2]]);
        $inputs = [
            $g->Variable($K->zeros([1,5,2])),
            $g->Variable($K->zeros([1,4,2])),
        ];
    
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Input shape is inconsistent: defined as [[3,2],[4,2]] but [[5,2],[4,2]] given in Attention');
        $layer->build($inputs);
    }

    public function testNormalForwardAndBackward()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new Attention($K);
        $inputs = [
            $g->Variable($K->zeros([2,2,3])),
            $g->Variable($K->zeros([2,4,3])),
        ];

        $layer->build($inputs);

        //
        // forward
        //
        //  batch size 2
        $query = $K->array([
            [[1,0,0],[0,1,0]],
            [[1,0,0],[0,1,0]],
        ]);
        $value = $K->array([
            [[1,0,0],[0,1,0],[0,0,1],[0,0,0]],
            [[1,0,0],[0,1,0],[0,0,1],[0,0,0]],
        ]);
        $inputs = [$query,$value];
        $copyInputs = [$K->copy($query),$K->copy($value)];
        [$outputsVariable,$scores] = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$inputs) {
                [$outputsVariable,$scores] = $layer->forward($inputs, $training=true,
                                ['return_attention_scores'=>true]);
                return [$outputsVariable,$scores];
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        //
        $this->assertEquals([2,2,4],$scores->shape());
        $this->assertEquals([2,2,3],$outputs->shape());
        $this->assertEquals($copyInputs[0]->toArray(),$inputs[0]->toArray());
        $this->assertEquals($copyInputs[1]->toArray(),$inputs[1]->toArray());
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->softmax($mo->array([
                [1,0,0,0],[0,1,0,0],
                [1,0,0,0],[0,1,0,0],
            ])),
            $K->ndarray($scores->reshape([4,4]))));
        $this->assertTrue($mo->la()->isclose(
            $mo->array([
                [[0.475367,0.174878,0.174878],[0.174878,0.475367,0.174878]],
                [[0.475367,0.174878,0.174878],[0.174878,0.475367,0.174878]],
            ]),
            $K->ndarray($outputs)));
        //
        // backward
        //
        // 2 batch
        $dOutputs = $K->ones($outputs->shape());

        $copydOutputs = $K->copy(
            $dOutputs);
        $dInputs = $outputsVariable->creator()->backward([$dOutputs]);
        // 2 batch
        $this->assertCount(2,$dInputs);
        $this->assertEquals([2,2,3],$dInputs[0]->shape());
        $this->assertEquals([2,4,3],$dInputs[1]->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());
        $this->assertTrue($mo->la()->isclose(
            $mo->array([
                [[0.08313105, 0.0305822 , 0.0305822],
                 [0.0305822 , 0.08313105, 0.0305822]],
                [[0.08313105, 0.0305822 , 0.0305822],
                 [0.0305822 , 0.08313105, 0.0305822]],
            ]),
            $K->ndarray($dInputs[0])));
        $this->assertTrue($mo->la()->isclose(
            $mo->array([
                [[0.73337567, 0.68082684, 0.65024465],
                 [0.68082684, 0.73337567, 0.65024465],
                 [0.38033763, 0.38033763, 0.34975544],
                 [0.20545992, 0.20545992, 0.34975544]],
                [[0.73337567, 0.68082684, 0.65024465],
                 [0.68082684, 0.73337567, 0.65024465],
                 [0.38033763, 0.38033763, 0.34975544],
                 [0.20545992, 0.20545992, 0.34975544]],
            ]),
            $K->ndarray($dInputs[1])));
    }
}
