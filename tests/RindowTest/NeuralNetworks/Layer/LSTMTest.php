<?php
namespace RindowTest\NeuralNetworks\Layer\LSTMTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Layer\LSTM;
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

    public function verifyGradient($mo, $K, $function, NDArray $x)
    {
        $f = function($x) use ($mo,$K,$function){
            $x = $K->array($x);
            $y = $function->forward($x,$training=true);
            return $K->ndarray($y);
        };
        $grads = $mo->la()->numericalGradient(1e-3,$f,$K->ndarray($x));
        $outputs = $function->forward($x,$training=true);
        $dOutputs = $K->ones($outputs->shape(),$outputs->dtype());
        $dInputs = $function->backward($dOutputs);
#echo "\n";
#echo "grads=".$mo->toString($grads[0],'%5.3f',true)."\n\n";
#echo "dInputs=".$mo->toString($dInputs,'%5.3f',true)."\n\n";
#echo $mo->asum($mo->op($grads[0],'-',$dInputs))."\n";
        return $mo->la()->isclose($grads[0],$K->ndarray($dInputs),null,1e-4);
    }

    public function testDefaultInitialize()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $layer = new LSTM(
            $backend,
            $units=4,
            [
                'input_shape'=>[5,3],
            ]);

        $layer->build();
        $params = $layer->getParams();
        $this->assertCount(3,$params);
        $this->assertEquals([3,4*4],$params[0]->shape());
        $this->assertEquals([4,4*4],$params[1]->shape());
        $this->assertEquals([4*4],$params[2]->shape());

        $grads = $layer->getGrads();
        $this->assertCount(3,$grads);
        $this->assertEquals([3,4*4],$grads[0]->shape());
        $this->assertEquals([4,4*4],$grads[1]->shape());
        $this->assertEquals([4*4],$grads[2]->shape());
        $this->assertNull(
            $layer->getActivation()
            );

        //$this->assertEquals([3],$layer->inputShape());
        $this->assertEquals([4],$layer->outputShape());
    }

    public function testNotspecifiedInputShape()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $layer = new LSTM(
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
        $layer = new LSTM(
            $backend,
            $units=4,
            [
            ]);
        $layer->build($inputShape=[5,3]);

        //$this->assertEquals([3],$layer->inputShape());
        $this->assertEquals([4],$layer->outputShape());
    }

    public function testInitializeWithReturnSequence()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $layer = new LSTM(
            $backend,
            $units=4,
            [
                'input_shape'=>[5,3],
                'return_sequences'=>true,
                'return_state'=>true,
            ]);
        $layer->build();

        //$this->assertEquals([3],$layer->inputShape());
        $this->assertEquals([5,4],$layer->outputShape());
    }

    public function testDefaultForwardAndBackward()
    {
        $mo = new MatrixOperator();
        $K = $backend = $this->newBackend($mo);
        $fn = $backend;

        $layer = new LSTM(
            $backend,
            $units=4,
            [
                'input_shape'=>[5,3],
            ]);

        $layer->build();
        $grads = $layer->getGrads();


        //
        // forward
        //
        //  2 batch
        $inputs = $K->ones([6,5,3]);
        $initialStates = [$K->ones([6,4]),$K->ones([6,4])];
        $copyInputs = $K->copy($inputs);
        $copyStates = [
            $K->copy($initialStates[0]),
            $K->copy($initialStates[1])];
        $outputs = $layer->forward($inputs,$training=true, $initialStates
        );
        //
        $this->assertEquals([6,4],$outputs->shape());
        $this->assertEquals($copyInputs->toArray(),$inputs->toArray());

        //
        // backward
        //
        // 2 batch
        $dOutputs =
            $K->ones([6,4]);
        $dStates =
            [$K->ones([6,4]),$K->ones([6,4])];

        $copydOutputs = $K->copy(
            $dOutputs);
        $copydStates = [
            $K->copy($dStates[0]),
            $K->copy($dStates[1])];
        $dInputs = $layer->backward($dOutputs,$dStates);
        // 2 batch
        $this->assertEquals([6,5,3],$dInputs->shape());
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
        $this->assertEquals($copydStates[1]->toArray(),$dStates[1]->toArray());
    }

    public function testForwardAndBackwardWithReturnSeqquence()
    {
        $mo = new MatrixOperator();
        $K = $backend = $this->newBackend($mo);
        $fn = $backend;

        $layer = new LSTM(
            $backend,
            $units=4,
            [
                'input_shape'=>[5,3],
                'return_sequences'=>true,
                'return_state'=>true,
            ]);

        $layer->build();
        $grads = $layer->getGrads();


        //
        // forward
        //
        //  2 batch
        $inputs = $K->ones([6,5,3]);
        $initialStates = [
            $K->ones([6,4]),
            $K->ones([6,4])];
        $copyInputs = $K->copy($inputs);
        $copyStates = [
            $K->copy($initialStates[0]),
            $K->copy($initialStates[1])];
        [$outputs,$nextStates] = $layer->forward($inputs,$training=true, $initialStates
        );
        //
        $this->assertEquals([6,5,4],$outputs->shape());
        $this->assertCount(2,$nextStates);
        $this->assertEquals([6,4],$nextStates[0]->shape());
        $this->assertEquals([6,4],$nextStates[1]->shape());
        $this->assertEquals($copyInputs->toArray(),$inputs->toArray());
        $this->assertEquals($copyStates[0]->toArray(),$initialStates[0]->toArray());
        $this->assertEquals($copyStates[1]->toArray(),$initialStates[1]->toArray());

        //
        // backward
        //
        // 2 batch
        $dOutputs =
            $K->ones([6,5,4]);
        $dStates = [
            $K->ones([6,4]),
            $K->ones([6,4])];

        $copydOutputs = $K->copy(
            $dOutputs);
        $copydStates = [
            $K->copy($dStates[0]),
            $K->copy($dStates[1])];
        [$dInputs,$dPrevStates] = $layer->backward($dOutputs,$dStates);
        // 2 batch
        $this->assertEquals([6,5,3],$dInputs->shape());
        $this->assertCount(2,$dPrevStates);
        $this->assertEquals([6,4],$dPrevStates[0]->shape());
        $this->assertEquals([6,4],$dPrevStates[1]->shape());
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
        $this->assertEquals($copydStates[1]->toArray(),$dStates[1]->toArray());
    }

    public function testForwardAndBackwardWithReturnSeqquenceWithoutInitialStates()
    {
        $mo = new MatrixOperator();
        $K = $backend = $this->newBackend($mo);
        $fn = $backend;

        $layer = new LSTM(
            $backend,
            $units=4,
            [
                'input_shape'=>[5,3],
                'return_sequences'=>true,
                'return_state'=>true,
            ]);

        $layer->build();
        $grads = $layer->getGrads();


        //
        // forward
        //
        //  2 batch
        $inputs = $K->ones([6,5,3]);
        $initialStates = null;
        $copyInputs = $K->copy($inputs);
        //$copyStates = [$mo->copy($initialStates[0])];
        [$outputs,$nextStates] = $layer->forward($inputs,$training=true, $initialStates
        );
        //
        $this->assertEquals([6,5,4],$outputs->shape());
        $this->assertCount(2,$nextStates);
        $this->assertEquals([6,4],$nextStates[0]->shape());
        $this->assertEquals([6,4],$nextStates[1]->shape());
        $this->assertEquals($copyInputs->toArray(),$inputs->toArray());
        //$this->assertEquals($copyStates[0]->toArray(),$initialStates[0]->toArray());

        //
        // backward
        //
        // 2 batch
        $dOutputs =
            $K->ones([6,5,4]);
        $dStates =
            [$K->ones([6,4]),
            $K->ones([6,4])];

        $copydOutputs = $K->copy(
            $dOutputs);
        $copydStates = [
            $K->copy($dStates[0]),
            $K->copy($dStates[1])];
        [$dInputs,$dPrevStates] = $layer->backward($dOutputs,$dStates);
        // 2 batch
        $this->assertEquals([6,5,3],$dInputs->shape());
        $this->assertCount(2,$dPrevStates);
        $this->assertEquals([6,4],$dPrevStates[0]->shape());
        $this->assertEquals([6,4],$dPrevStates[1]->shape());
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
        $this->assertEquals($copydStates[1]->toArray(),$dStates[1]->toArray());
    }

    public function testOutputsAndGrads()
    {
        $mo = new MatrixOperator();
        $K = $backend = $this->newBackend($mo);
        $fn = $backend;

        $layer = new LSTM(
            $backend,
            $units=4,
            [
                'input_shape'=>[3,5],
                'return_sequences'=>true,
                'return_state'=>true,
                'activation'=>null,
                'recurrent_activation'=>null,
            ]);

        $kernel = $K->ones([5,4*4]);
        $recurrent = $K->ones([4,4*4]);
        $bias = $K->ones([4*4]);
        $layer->build(null,
            ['sampleWeights'=>[$kernel,$recurrent,$bias]]
        );
        $this->assertNull($layer->getActivation());
        $grads = $layer->getGrads();


        //
        // forward
        //
        //  2 batch
        $inputs = $K->ones([2,3,5]);
        $states = [
            $K->ones([2,4]),
            $K->ones([2,4])];
        [$outputs,$nextStates] = $layer->forward($inputs,$training=true, $states);
        //
        $this->assertEquals(
            [2,3,4],
            $outputs->shape());
        $this->assertEquals(
            [2,4],
            $nextStates[0]->shape());
        $this->assertEquals(
            [2,4],
            $nextStates[1]->shape());
        //
        // backward
        //
        // 2 batch
        $dOutputs =
            $K->ones([2,3,4]);
        $dStates =
            [$K->ones([2,4]),
            $K->ones([2,4])];

        [$dInputs,$dPrevStates] = $layer->backward($dOutputs,$dStates);
        // 2 batch
        $this->assertEquals(
            [2,3,5],
            $dInputs->shape());
        $this->assertEquals(
            [2,4],
            $dPrevStates[0]->shape());
        $this->assertEquals(
            [2,4],
            $dPrevStates[1]->shape());
        $this->assertEquals(
            [5,16],
            $grads[0]->shape());
        $this->assertEquals(
            [4,16],
            $grads[1]->shape());
        $this->assertEquals(
            [16],
            $grads[2]->shape());
    }

    public function testVerifyReturnSequences()
    {
        $mo = new MatrixOperator();
        $K = $backend = $this->newBackend($mo);
        $fn = $backend;

        $layer = new LSTM(
            $backend,
            $units=3,
            [
                'input_shape'=>[4,10],
                'return_sequences'=>true,
                #'return_state'=>true,
                #'activation'=>null,
            ]);
        $layer->build();
        $weights = $layer->getParams();

        $x = $K->array([
            [0,1,2,9],
        ]);
        $x = $K->onehot($x->reshape([4]),$numClass=10)->reshape([1,4,10]);
        $outputs = $layer->forward($x,$training=true);

        $this->assertTrue(
            $this->verifyGradient($mo,$K,$layer,$x));
    }
}
