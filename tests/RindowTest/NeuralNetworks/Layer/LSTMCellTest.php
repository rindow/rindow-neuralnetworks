<?php
namespace RindowTest\NeuralNetworks\Layer\LSTMCellTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Layer\LSTMCell;
use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Activation\Tanh;

class LSTMCellTest extends TestCase
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
            $object = new \stdClass();
            $x = $K->array($x);
            [$y,$states] = $function->forward($x,$states,calcState:$object);
            return $K->ndarray($y);
        };
        $grads = $mo->la()->numericalGradient(1e-3,$f,$K->ndarray($x));
        $object = new \stdClass();
        $next_states = $function->forward($x,$states,calcState:$object);
        $dNextStates = [$K->ones([1,3]),$K->zeros([1,3])];
        [$dInputs,$dPrevStates] = $function->backward($dNextStates,$object);

        return $mo->la()->isclose($grads[0],$K->ndarray($dInputs),null,1e-4);
    }

    public function testDefaultInitialize()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new LSTMCell(
            $K,
            $units=4,
            input_shape:[3]
            );

        $layer->build([3]);
        $params = $layer->getParams();
        $this->assertCount(3,$params);
        $this->assertEquals([3,16],$params[0]->shape());
        $this->assertEquals([4,16],$params[1]->shape());
        $this->assertEquals([16],$params[2]->shape());

        $grads = $layer->getGrads();
        $this->assertCount(3,$grads);
        $this->assertEquals([3,16],$grads[0]->shape());
        $this->assertEquals([4,16],$grads[1]->shape());
        $this->assertEquals([16],$grads[2]->shape());

        //$this->assertEquals([3],$layer->inputShape());
        $this->assertEquals([4],$layer->outputShape());
    }

    public function testSetInputShape()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new LSTMCell(
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
        $layer = new LSTMCell(
            $K,
            $units=4,
            input_shape:[3],
            );

        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Input shape is inconsistent: defined as (3) but (4) given in LSTMCell');
        $layer->build([4]);
    }

    public function testNormalForwardAndBackward()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $fn = $K;

        $layer = new LSTMCell(
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
        $states = [$K->ones([2,4]),$K->ones([2,4])];
        $object = new \stdClass();
        $copyInputs = $K->copy($inputs);
        $copyStates = [
            $K->copy($states[0]),
            $K->copy($states[1])];
        $nextStates = $layer->forward($inputs, $states,calcState:$object);
        //
        $this->assertCount(2,$nextStates);
        $this->assertEquals([2,4],$nextStates[0]->shape());
        $this->assertEquals([2,4],$nextStates[1]->shape());
        $this->assertEquals($copyInputs->toArray(),$inputs->toArray());
        $this->assertEquals($copyStates[0]->toArray(),$states[0]->toArray());
        $this->assertEquals($copyStates[1]->toArray(),$states[1]->toArray());

        //
        // backward
        //
        // 2 batch
        $dStates =
            [$K->ones([2,4]),$K->ones([2,4])];

        $copydStates = [
            $K->copy($dStates[0]),
            $K->copy($dStates[1])];
        [$dInputs,$dPrevStates] = $layer->backward($dStates,$object);
        // 2 batch
        $this->assertEquals([2,3],$dInputs->shape());
        $this->assertCount(2,$dPrevStates);
        $this->assertEquals([2,4],$dPrevStates[0]->shape());
        $this->assertEquals([2,4],$dPrevStates[1]->shape());
        $this->assertNotEquals(
            $mo->zerosLike($grads[0])->toArray(),
            $grads[0]->toArray());
        $this->assertNotEquals(
            $mo->zerosLike($grads[1])->toArray(),
            $grads[1]->toArray());
        $this->assertNotEquals(
            $mo->zerosLike($grads[2])->toArray(),
            $grads[2]->toArray());

        $this->assertEquals($copydStates[0]->toArray(),$dStates[0]->toArray());
        $this->assertEquals($copydStates[1]->toArray(),$dStates[1]->toArray());
    }

    public function testOutputsAndGrads()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $fn = $K;

        $layer = new LSTMCell(
            $K,
            $units=4,
            input_shape:[3],
            activation:'linear',
            recurrent_activation:'linear',
            );

        $kernel = $K->ones([3,4*4]);
        $recurrent = $K->ones([4,4*4]);
        $bias = $K->ones([4*4]);
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
        $states = [$K->ones([2,4]),$K->ones([2,4])];
        $object = new \stdClass();
        $nextStates = $layer->forward($inputs, $states,calcState:$object);
        //
        $this->assertEquals([
            [576,576,576,576],
            [576,576,576,576],
            ],$nextStates[0]->toArray());
        $this->assertEquals([
            [72,72,72,72],
            [72,72,72,72],
            ],$nextStates[1]->toArray());
        //
        // backward
        //
        // 2 batch
        $dStates =
            [$K->scale(2,$K->ones([2,4])),$K->ones([2,4])];

        [$dInputs,$dPrevStates] = $layer->backward($dStates,$object);
        // 2 batch
        $this->assertEquals([
            [1732,1732,1732],
            [1732,1732,1732],
            ],$dInputs->toArray());
        $this->assertEquals([
            [1732,1732,1732,1732],
            [1732,1732,1732,1732],
            ],$dPrevStates[0]->toArray());
        $this->assertEquals([
            [136,136,136,136],
            [136,136,136,136],
            ],$dPrevStates[1]->toArray());
        $this->assertEquals([
            [272,272,272,272,
             34,34,34,34,
             272,272,272,272,
             288,288,288,288],
            [272,272,272,272,
             34,34,34,34,
             272,272,272,272,
             288,288,288,288],
            [272,272,272,272,
             34,34,34,34,
             272,272,272,272,
             288,288,288,288],
            ],$grads[0]->toArray());
        $this->assertEquals([
            [272,272,272,272,
             34,34,34,34,
             272,272,272,272,
             288,288,288,288],
            [272,272,272,272,
             34,34,34,34,
             272,272,272,272,
             288,288,288,288],
            [272,272,272,272,
             34,34,34,34,
             272,272,272,272,
             288,288,288,288],
            [272,272,272,272,
             34,34,34,34,
             272,272,272,272,
             288,288,288,288],
            ],$grads[1]->toArray());
        $this->assertEquals(
            [272,272,272,272,
             34,34,34,34,
             272,272,272,272,
             288,288,288,288],
            $grads[2]->toArray());
    }

    public function testVerifyGradient()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $fn = $K;

        $layer = new LSTMCell(
            $K,
            $units=3,
                input_shape:[10],
                #activation:'linear',
            );
        $layer->build([10]);
        $weights = $layer->getParams();

        $x = $K->array([
            [1],
        ],dtype:NDArray::int32);
        $states = [$K->zeros([1,3]),$K->zeros([1,3])];
        $object = new \stdClass();
        $x = $K->onehot($x->reshape([1]),$numClass=10)->reshape([1,10]);
        $outputs = $layer->forward($x,$states,calcState:$object);

        $this->assertTrue(
            $this->verifyGradient($mo,$K,$layer,$x,$states));
    }
}
