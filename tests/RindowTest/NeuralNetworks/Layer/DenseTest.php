<?php
namespace RindowTest\NeuralNetworks\Layer\DenseTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Layer\Dense;
use Rindow\NeuralNetworks\Gradient\Core\Undetermined;
use Rindow\NeuralNetworks\Gradient\Core\UndeterminedNDArray;
use InvalidArgumentException;

class Test extends TestCase
{
    public function newMatrixOperator()
    {
        return new MatrixOperator();
    }

    public function newBackend($mo)
    {
        $builder = new NeuralNetworks($mo);
        return $builder->backend();
    }

    public function newInputShape($inputShape)
    {
        array_unshift($inputShape,1);
        $variable = new Undetermined(new UndeterminedNDArray($inputShape));
        return $variable;
    }

    public function testDefaultInitialize()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $layer = new Dense($K,$units=3,['input_shape'=>[2]]);

        $layer->build();
        $params = $layer->getParams();
        $this->assertCount(2,$params);
        $this->assertEquals([2,3],$params[0]->shape());
        $this->assertEquals([3],$params[1]->shape());
        $this->assertNotEquals($mo->zeros([2,3])->toArray(),$params[0]->toArray());
        $this->assertEquals($mo->zeros([3])->toArray(),$params[1]->toArray());

        $grads = $layer->getGrads();
        $this->assertCount(2,$grads);
        $this->assertEquals([2,3],$grads[0]->shape());
        $this->assertEquals([3],$grads[1]->shape());
        $this->assertEquals($mo->zeros([2,3])->toArray(),$grads[0]->toArray());
        $this->assertEquals($mo->zeros([3])->toArray(),$grads[1]->toArray());

        $this->assertEquals([3],$layer->outputShape());
        //$layer->unlink();
    }

    public function testNotspecifiedInputShape()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $layer = new Dense($K,$units=3,[]);

        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Input shape is not defined');
        $layer->build();
    }

    public function testSetInputShape()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $layer = new Dense($K,$units=3,[]);
        $layer->build($this->newInputShape([2]));
        $params = $layer->getParams();
        $this->assertCount(2,$params);
        $this->assertEquals([2,3],$params[0]->shape());

        $this->assertEquals([3],$layer->outputShape());
    }

    public function testNormalForwardAndBackward()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $fn = $mo->la();

        $layer = new Dense($K,$units=2,['input_shape'=>[3]]);

        $layer->build(null,[
            'sampleWeights'=>[
                $K->array([[0.1, 0.2], [0.1, 0.1], [0.2, 0.2]]), // kernel
                $K->array([0.5, 0.1]),                         // bias
            ]
        ]);

        //
        // forward
        //
        // 3 input x 4 minibatch
        $inputs = $mo->array([
            [0.0, 0.0 , 6.0],
            [0.0, 0.0 , 6.0],
            [0.0, 0.0 , 6.0],
            [0.0, 0.0 , 6.0],
        ]);
        $copyInputs = $mo->copy($inputs);
        $inputs = $K->array($inputs);
        $outputs = $layer->forward($inputs, $training=true);
        $outputs = $K->ndarray($outputs);
        $inputs = $K->ndarray($inputs);
        // 2 output x 4 batch
        $this->assertEquals([4,2],$outputs->shape());
        $this->assertTrue($fn->isclose($mo->array([
                [1.7, 1.3],
                [1.7, 1.3],
                [1.7, 1.3],
                [1.7, 1.3],
            ]),$outputs
        ));
        $this->assertEquals($copyInputs->toArray(),$inputs->toArray());

        //
        // backward
        //
        // 2 output x 4 batch
        $dOutputs = $mo->array([
            [0.0, -0.5],
            [0.0, -0.5],
            [0.0, -0.5],
            [0.0, -0.5],
        ]);
        $copydOutputs = $mo->copy($dOutputs);
        $dOutputs = $K->array($dOutputs);
        [$dInputs] = $layer->backward([$dOutputs]);
        $dInputs = $K->ndarray($dInputs);
        $dOutputs = $K->ndarray($dOutputs);
        // 3 input x 4 batch
        $this->assertEquals([4,3],$dInputs->shape());
        $this->assertTrue($fn->isclose($mo->array([
                [-0.1, -0.05 , -0.1],
                [-0.1, -0.05 , -0.1],
                [-0.1, -0.05 , -0.1],
                [-0.1, -0.05 , -0.1],
            ]),$dInputs
        ));
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());
    }

    public function testNdInput()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $layer = new Dense($K,$units=4,['input_shape'=>[2,3]]);

        $layer->build();
        $params = $layer->getParams();
        $this->assertCount(2,$params);
        $this->assertEquals([3,4],$params[0]->shape());
        $this->assertEquals([4],$params[1]->shape());
        $this->assertNotEquals($mo->zeros([2,4])->toArray(),$params[0]->toArray());
        $this->assertEquals($mo->zeros([4])->toArray(),$params[1]->toArray());

        $grads = $layer->getGrads();
        $this->assertCount(2,$grads);
        $this->assertEquals([3,4],$grads[0]->shape());
        $this->assertEquals([4],$grads[1]->shape());
        $this->assertEquals($mo->zeros([3,4])->toArray(),$grads[0]->toArray());
        $this->assertEquals($mo->zeros([4])->toArray(),$grads[1]->toArray());

        $this->assertEquals([2,4],$layer->outputShape());

        $inputs = $mo->zeros([10,2,3]);
        $inputs = $K->array($inputs);
        $outputs = $layer->forward($inputs,true);
        $outputs = $K->ndarray($outputs);
        $this->assertEquals([10,2,4],$outputs->shape());
    }
}
