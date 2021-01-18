<?php
namespace RindowTest\NeuralNetworks\Layer\ConcatenateTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Layer\Concatenate;
use InvalidArgumentException;

class Test extends TestCase
{
    public function newBackend($mo)
    {
        $builder = new NeuralNetworks($mo);
        return $builder->backend();
    }

    public function testDefaultInitialize()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $layer = new Concatenate(
            $backend,
            [
                #'axis'=>-1,
                'input_shapes'=>[[4,3],[4,2]],
            ]);

        $layer->build();
        $params = $layer->getParams();
        $this->assertCount(0,$params);

        $grads = $layer->getGrads();
        $this->assertCount(0,$grads);

        $this->assertEquals([4,5],$layer->outputShape());
    }

    public function testNotspecifiedInputShape()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $layer = new Concatenate(
            $backend,
            [
            ]);

        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Input shape is not defined');
        $layer->build();
    }

    public function testSetInputShape()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $layer = new Concatenate(
            $backend,
            [
                'axis'=>1,
            ]);
        // [batch,2,4],[batch,3,4]
        $layer->build($inputShape=[[2,4],[3,4]]);
        // [batch,5,4]
        $this->assertEquals([5,4],$layer->outputShape());
    }

    public function testNormalForwardAndBackward()
    {
        $mo = new MatrixOperator();
        $K = $backend = $this->newBackend($mo);
        $fn = $backend;

        $layer = new Concatenate(
            $backend,
            [
                #'axis'=>-1,
            ]);

        $layer->build($inputShape=[[2,2],[2,3]]);

        //
        // forward
        //
        //  batch size 2
        $i1 = $K->array($mo->arange(2*2*2,null,null,NDArray::float32)->reshape([2,2,2]));
        $i2 = $K->array($mo->arange(2*2*3,100,null,NDArray::float32)->reshape([2,2,3]));
        $inputs = [$i1,$i2];
        $copyInputs = [$K->copy($i1),$K->copy($i2)];
        $outputs = $layer->forward($inputs, $training=true);
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
        $dOutputs = $K->copy($outputs);

        $copydOutputs = $K->copy(
            $dOutputs);
        $dInputs = $layer->backward($dOutputs);
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
