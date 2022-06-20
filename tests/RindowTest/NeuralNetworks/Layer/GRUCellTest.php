<?php
namespace RindowTest\NeuralNetworks\Layer\GRUCellTest;

use InvalidArgumentException;
use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Layer\GRUCell;
use Rindow\NeuralNetworks\Activation\Tanh;

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

        return $mo->la()->isclose($grads[0],$K->ndarray($dInputs),1e-3);
    }

    public function testDefaultInitialize()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new GRUCell(
            $K,
            $units=4,
            input_shape:[3]
            );

        $layer->build([3]);
        $params = $layer->getParams();
        $this->assertCount(3,$params);
        $this->assertEquals([3,12],$params[0]->shape());
        $this->assertEquals([4,12],$params[1]->shape());
        $this->assertEquals([2,12],$params[2]->shape());

        $grads = $layer->getGrads();
        $this->assertCount(3,$grads);
        $this->assertEquals([3,12],$grads[0]->shape());
        $this->assertEquals([4,12],$grads[1]->shape());
        $this->assertEquals([2,12],$grads[2]->shape());
        $this->assertInstanceOf(
            Tanh::class, $layer->getActivation()
        );

        //$this->assertEquals([3],$layer->inputShape());
        $this->assertEquals([4],$layer->outputShape());
    }

    public function testInitializeWithoutResetAfter()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new GRUCell(
            $K,
            $units=4,
            input_shape:[3],
            reset_after:false,
            );

        $layer->build([3]);
        $params = $layer->getParams();
        $this->assertCount(3,$params);
        $this->assertEquals([3,12],$params[0]->shape());
        $this->assertEquals([12,4],$params[1]->shape());
        $this->assertEquals([12],$params[2]->shape());

        $grads = $layer->getGrads();
        $this->assertCount(3,$grads);
        $this->assertEquals([3,12],$grads[0]->shape());
        $this->assertEquals([12,4],$grads[1]->shape());
        $this->assertEquals([12],$grads[2]->shape());

        //$this->assertEquals([3],$layer->inputShape());
        $this->assertEquals([4],$layer->outputShape());
    }

    public function testSetInputShape()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new GRUCell(
            $K,
            $units=4,
            );
        $layer->build($inputShape=[3]);

        //$this->assertEquals([3],$layer->inputShape());
        $this->assertEquals([4],$layer->outputShape());
    }

    public function testUnmatchSpecifiedInputShape()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new GRUCell(
            $K,
            $units=4,
            input_shape:[3],
            );

        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Input shape is inconsistent: defined as [3] but [4] given in GRUCell');
        $layer->build([4]);
    }

    public function testNormalForwardAndBackward()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $fn = $K;

        $layer = new GRUCell(
            $K,
            $units=4,
            input_shape:[3]
            );

        $layer->build([3]);
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
        $copydStates = [$K->copy($dStates[0])];
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

    public function testOutputsAndGradsWithResetAfter()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $fn = $K;

        $layer = new GRUCell(
            $K,
            $units=4,
            input_shape:[3],
            activation:'linear',
            recurrent_activation:'linear',
            );

        $kernel = $K->ones([3,4*3]);
        $recurrent = $K->ones([4,4*3]);
        $bias = $K->ones([2,4*3]);
        $layer->build([3],
            sampleWeights:[$kernel,$recurrent,$bias]
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
            [-383,-383,-383,-383],
            [-383,-383,-383,-383],
            ],$outputs->toArray());
        $this->assertEquals([
            [-383,-383,-383,-383],
            [-383,-383,-383,-383],
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
            [-3328,-3328,-3328],
            [-3328,-3328,-3328],
            ],$dInputs->toArray());
        $this->assertEquals([
            [-3822,-3822,-3822,-3822],
            [-3822,-3822,-3822,-3822],
            ],$dPrevStates[0]->toArray());
        $this->assertEquals([
            [-192,-192,-192,-192,
             -1440,-1440,-1440,-1440,
             -32,-32,-32,-32],
            [-192,-192,-192,-192,
             -1440,-1440,-1440,-1440,
             -32,-32,-32,-32],
            [-192,-192,-192,-192,
             -1440,-1440,-1440,-1440,
             -32,-32,-32,-32],
            ],$grads[0]->toArray());
        $this->assertEquals([
            [-192,-192,-192,-192,
             -1440,-1440,-1440,-1440,
             -288,-288,-288,-288],
            [-192,-192,-192,-192,
             -1440,-1440,-1440,-1440,
             -288,-288,-288,-288],
            [-192,-192,-192,-192,
             -1440,-1440,-1440,-1440,
             -288,-288,-288,-288],
            [-192,-192,-192,-192,
             -1440,-1440,-1440,-1440,
             -288,-288,-288,-288],
            ],$grads[1]->toArray());
        $this->assertEquals([
            [-192,-192,-192,-192,
             -1440,-1440,-1440,-1440,
             -32,-32,-32,-32],
            [-192,-192,-192,-192,
             -1440,-1440,-1440,-1440,
             -288,-288,-288,-288],
         ],$grads[2]->toArray());
    }

    public function testOutputsAndGradsWithoutResetAfter()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $fn = $K;

        $layer = new GRUCell(
            $K,
            $units=4,
            input_shape:[3],
            activation:'linear',
            recurrent_activation:'linear',
            reset_after:false,
            );

        $kernel = $K->ones([3,4*3]);
        $recurrent = $K->ones([4*3,4]);
        $bias = $K->ones([4*3]);
        $layer->build([3],
            sampleWeights:[$kernel,$recurrent,$bias]
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
            [-244,-244,-244,-244],
            [-244,-244,-244,-244],
            ],$outputs->toArray());
        $this->assertEquals([
            [-244,-244,-244,-244],
            [-244,-244,-244,-244],
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
            [-560,-560,-560],
            [-560,-560,-560],
            ],$dInputs->toArray());
        $this->assertEquals([
            [-936,-936,-936,-936],
            [-936,-936,-936,-936],
            ],$dPrevStates[0]->toArray());
        $this->assertEquals([
            [-140,-140,-140,-140,
             -112,-112,-112,-112,
             -28,-28,-28,-28],
            [-140,-140,-140,-140,
             -112,-112,-112,-112,
             -28,-28,-28,-28],
            [-140,-140,-140,-140,
             -112,-112,-112,-112,
             -28,-28,-28,-28],
            ],$grads[0]->toArray());
        $this->assertEquals([
            [-140,-140,-140,-140],
            [-140,-140,-140,-140],
            [-140,-140,-140,-140],
            [-140,-140,-140,-140],
            [-112,-112,-112,-112],
            [-112,-112,-112,-112],
            [-112,-112,-112,-112],
            [-112,-112,-112,-112],
            [-224,-224,-224,-224],
            [-224,-224,-224,-224],
            [-224,-224,-224,-224],
            [-224,-224,-224,-224],
            ],$grads[1]->toArray());
        $this->assertEquals(
            [-140,-140,-140,-140,
             -112,-112,-112,-112,
             -28,-28,-28,-28],
            $grads[2]->toArray());
    }

    public function testVerifyGradientResetAfter()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $fn = $K;

        $layer = new GRUCell(
            $K,
            $units=3,
            input_shape:[10],
            #activation:'linear',
            );
        $layer->build([10]);
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

    public function testVerifyGradientWithoutResetAfter()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $fn = $K;

        $layer = new GRUCell(
            $K,
            $units=3,
            input_shape:[10],
            #activation:'linear',
            reset_after:false,
            );
        $layer->build([10]);
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
