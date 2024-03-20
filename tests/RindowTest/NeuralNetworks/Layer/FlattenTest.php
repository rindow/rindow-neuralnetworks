<?php
namespace RindowTest\NeuralNetworks\Layer\FlattenTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Layer\Flatten;
use InvalidArgumentException;

class FlattenTest extends TestCase
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
        $layer = new Flatten(
            $K,input_shape:[4,4,3]);
        $inputs = $g->Variable($K->zeros([1,4,4,3]));
        $layer->build($inputs);
        $params = $layer->getParams();
        $this->assertCount(0,$params);

        $grads = $layer->getGrads();
        $this->assertCount(0,$grads);

        $this->assertEquals([48],$layer->outputShape());
    }

    public function testSetInputShape()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();

        $layer = new Flatten(
            $K,
            );
        
        $inputs = $g->Variable($K->zeros([1,4,4,3]));
        $layer->build($inputs);

        $this->assertEquals([48],$layer->outputShape());
    }

    public function testUnmatchSpecifiedInputShape()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new Flatten(
            $K,
            input_shape:[4,4,3]
            );
        $inputs = $g->Variable($K->zeros([1,4,4,5]));
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Input shape is inconsistent: defined as [4,4,3] but [4,4,5] given in Flatten');
        $layer->build($inputs);
    }

    public function testNormalForwardAndBackward()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $fn = $K;

        $layer = new Flatten(
            $K,
            input_shape:[4,4,3]);

        //$layer->build($g->Variable($inputs));

        //
        // forward
        //
        //  batch size 2
        $inputs = $K->array($mo->arange(2*4*4*3)->reshape([2,4,4,3]));
        $copyInputs = $K->copy($inputs);
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$inputs) {
                $outputsVariable = $layer->forward($inputs);
                return $outputsVariable;
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        //
        $this->assertEquals(
            [2,48],$outputs->shape());
        $this->assertEquals($copyInputs->toArray(),$inputs->toArray());

        //
        // backward
        //
        // 2 batch
        $dOutputs =
            $K->array($mo->arange(2*4*4*3)->reshape([2,4*4*3]));

        $copydOutputs = $K->copy(
            $dOutputs);
        [$dInputs] = $outputsVariable->creator()->backward([$dOutputs]);
        // 2 batch
        $this->assertEquals([2,4,4,3],$dInputs->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());
    }
}
