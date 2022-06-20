<?php
namespace RindowTest\NeuralNetworks\Layer\SimpleRNNCellTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Layer\SimpleRNNCell;
use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Activation\Tanh;

class Test extends TestCase
{
    public function newBackend($mo)
    {
        $builder = new NeuralNetworks($mo);
        return $builder->backend();
    }

    public function verifyGradient($mo, $K, $function, NDArray $x,array $states)
    {
        $f = function($x) use ($mo,$K,$function,$states){
            $x = $K->array($x);
            $object = new \stdClass();
            [$y,$states] = $function->forward($x,$states,$training=true,$object);
            return $K->ndarray($y);
        };
        $grads = $mo->la()->numericalGradient(1e-3,$f,$K->ndarray($x));
        $object = new \stdClass();
        [$outputs,$next_states] = $function->forward($x,$states,$training=true,$object);
        $dOutputs = $K->ones($outputs->shape(),$outputs->dtype());
        $dNextStates = [$K->zeros([1,3])];
        [$dInputs,$dPrevStates] = $function->backward($dOutputs,$dNextStates,$object);

        return $mo->la()->isclose($grads[0],$K->ndarray($dInputs),1e-4);
    }

    public function testDefaultInitialize()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $layer = new SimpleRNNCell(
            $backend,
            $units=4,
            [
                'input_shape'=>[3]
            ]);

        $layer->build();
        $params = $layer->getParams();
        $this->assertCount(3,$params);
        $this->assertEquals([3,4],$params[0]->shape());
        $this->assertEquals([4,4],$params[1]->shape());
        $this->assertEquals([4],$params[2]->shape());

        $grads = $layer->getGrads();
        $this->assertCount(3,$grads);
        $this->assertEquals([3,4],$grads[0]->shape());
        $this->assertEquals([4,4],$grads[1]->shape());
        $this->assertEquals([4],$grads[2]->shape());
        $this->assertInstanceOf(
            Tanh::class, $layer->getActivation()
            );

        //$this->assertEquals([3],$layer->inputShape());
        $this->assertEquals([4],$layer->outputShape());
    }

    public function testNotspecifiedInputShape()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $layer = new SimpleRNNCell(
            $backend,
            $units=4,
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
        $layer = new SimpleRNNCell(
            $backend,
            $units=4,
            [
            ]);
        $layer->build($inputShape=[3]);

        //$this->assertEquals([3],$layer->inputShape());
        $this->assertEquals([4],$layer->outputShape());
    }

    public function testNormalForwardAndBackward()
    {
        $mo = new MatrixOperator();
        $K = $backend = $this->newBackend($mo);
        $fn = $backend;

        $layer = new SimpleRNNCell(
            $backend,
            $units=4,
            [
                'input_shape'=>[3]
            ]);

        $layer->build();
        $grads = $layer->getGrads();


        //
        // forward
        //
        //  2 batch
        $inputs = $K->ones([2,3]);
        $states = [$K->ones([2,4])];
        $object = new \stdClass();
        $copyInputs = $K->copy($inputs);
        $copyStates = [$K->copy($states[0])];
        [$outputs,$nextStates] = $layer->forward($inputs, $states,$training=true,$object);
        //
        $this->assertEquals([2,4],$outputs->shape());
        $this->assertCount(1,$nextStates);
        $this->assertEquals([2,4],$nextStates[0]->shape());
        $this->assertEquals($copyInputs->toArray(),$inputs->toArray());
        $this->assertEquals($copyStates[0]->toArray(),$states[0]->toArray());

        //
        // backward
        //
        // 2 batch
        $dOutputs =
            $K->ones([2,4]);
        $dStates =
            [$K->ones([2,4])];

        $copydOutputs = $K->copy(
            $dOutputs);
        $copydStates = [$K->copy(
            $dStates[0])];
        [$dInputs,$dPrevStates] = $layer->backward($dOutputs,$dStates,$object);
        // 2 batch
        $this->assertEquals([2,3],$dInputs->shape());
        $this->assertCount(1,$dPrevStates);
        $this->assertEquals([2,4],$dPrevStates[0]->shape());
        $this->assertNotEquals(
            $mo->zerosLike($grads[0])->toArray(),
            $grads[0]->toArray());
        $this->assertNotEquals(
            $mo->zerosLike($grads[1])->toArray(),
            $grads[1]->toArray());
        $this->assertNotEquals(
            $mo->zerosLike($grads[2])->toArray(),
            $grads[2]->toArray());

        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());
        $this->assertEquals($copydStates[0]->toArray(),$dStates[0]->toArray());
    }

    public function testOutputsAndGrads()
    {
        $mo = new MatrixOperator();
        $K = $backend = $this->newBackend($mo);
        $fn = $backend;

        $layer = new SimpleRNNCell(
            $backend,
            $units=4,
            [
                'input_shape'=>[3],
                'activation'=>null,
            ]);

        $kernel = $K->ones([3,4]);
        $recurrent = $K->ones([4,4]);
        $bias = $K->ones([4]);
        $layer->build(null,
            ['sampleWeights'=>[$kernel,$recurrent,$bias]]
        );
        $this->assertNull($layer->getActivation());
        $grads = $layer->getGrads();


        //
        // forward
        //
        //  2 batch
        $inputs = $K->ones([2,3]);
        $states = [$K->ones([2,4])];
        $object = new \stdClass();
        [$outputs,$nextStates] = $layer->forward($inputs, $states,$training=true,$object);
        //
        $this->assertEquals([
            [8,8,8,8],
            [8,8,8,8],
            ],$outputs->toArray());
        $this->assertEquals([
            [8,8,8,8],
            [8,8,8,8],
            ],$nextStates[0]->toArray());
        //
        // backward
        //
        // 2 batch
        $dOutputs =
            $K->ones([2,4]);
        $dStates =
            [$K->ones([2,4])];

        [$dInputs,$dPrevStates] = $layer->backward($dOutputs,$dStates,$object);
        // 2 batch
        $this->assertEquals([
            [8,8,8],
            [8,8,8],
            ],$dInputs->toArray());
        $this->assertEquals([
            [8,8,8,8],
            [8,8,8,8],
            ],$dPrevStates[0]->toArray());
        $this->assertEquals([
            [4,4,4,4],
            [4,4,4,4],
            [4,4,4,4],
            ],$grads[0]->toArray());
        $this->assertEquals([
            [4,4,4,4],
            [4,4,4,4],
            [4,4,4,4],
            [4,4,4,4],
            ],$grads[1]->toArray());
        $this->assertEquals(
            [4,4,4,4]
            ,$grads[2]->toArray());
    }

    public function testVerifyGradient()
    {
        $mo = new MatrixOperator();
        $K = $backend = $this->newBackend($mo);
        $fn = $backend;

        $layer = new SimpleRNNCell(
            $backend,
            $units=3,
            [
                'input_shape'=>[10],
                #'activation'=>null,
            ]);
        $layer->build();
        $weights = $layer->getParams();

        $x = $K->array([
            [1],
        ]);
        $states = [$K->zeros([1,3])];
        $object = new \stdClass();
        $x = $K->onehot($x->reshape([1]),$numClass=10)->reshape([1,10]);
        $outputs = $layer->forward($x,$states,$training=true,$object);

        $this->assertTrue(
            $this->verifyGradient($mo,$K,$layer,$x,$states));
    }
}
