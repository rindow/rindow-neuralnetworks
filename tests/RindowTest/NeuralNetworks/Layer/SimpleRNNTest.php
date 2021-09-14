<?php
namespace RindowTest\NeuralNetworks\Layer\SimpleRNNTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Layer\SimpleRNN;
use Rindow\NeuralNetworks\Gradient\Core\Undetermined;
use Rindow\NeuralNetworks\Gradient\Core\UndeterminedNDArray;
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

    public function newInputShape($inputShape)
    {
        array_unshift($inputShape,1);
        $variable = new Undetermined(new UndeterminedNDArray($inputShape));
        return $variable;
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
        [$dInputs] = $function->backward([$dOutputs]);
        return $mo->la()->isclose($grads[0],$K->ndarray($dInputs),1e-3);
    }

    public function testDefaultInitialize()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $layer = new SimpleRNN(
            $backend,
            $units=4,
            [
                'input_shape'=>[5,3],
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
        $layer = new SimpleRNN(
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
        $layer = new SimpleRNN(
            $backend,
            $units=4,
            [
            ]);
        $layer->build([$this->newInputShape([5,3])]);

        //$this->assertEquals([3],$layer->inputShape());
        $this->assertEquals([4],$layer->outputShape());
    }

    public function testSetInputShapeForSequential()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $layer = new SimpleRNN(
            $backend,
            $units=4,
            [
            ]);
        $layer->build($this->newInputShape([5,3]));

        //$this->assertEquals([3],$layer->inputShape());
        $this->assertEquals([4],$layer->outputShape());
    }

    public function testInitializeWithReturnSequence()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $layer = new SimpleRNN(
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

        $layer = new SimpleRNN(
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
        $initialStates = [$K->ones([6,4])];
        $copyInputs = $K->copy($inputs);
        $copyStates = [$K->copy($initialStates[0])];
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
            [$K->ones([6,4])];

        $copydOutputs = $K->copy(
            $dOutputs);
        $copydStates = [$K->copy(
            $dStates[0])];
        //$dInputs = $layer->backward($dOutputs,$dStates);
        $dPrevStates = $layer->backward([$dOutputs]);
        $dInputs = array_shift($dPrevStates);
        $this->assertCount(1,$dPrevStates);
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
    }

    public function testForwardAndBackwardWithReturnSeqquence()
    {
        $mo = new MatrixOperator();
        $K = $backend = $this->newBackend($mo);
        $fn = $backend;

        $layer = new SimpleRNN(
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
        $initialStates = [$K->ones([6,4])];
        $copyInputs = $K->copy($inputs);
        $copyStates = [$K->copy($initialStates[0])];
        [$outputs,$nextStates] = $layer->forward($inputs,$training=true, $initialStates
        );
        //
        $this->assertEquals([6,5,4],$outputs->shape());
        $this->assertCount(1,$nextStates);
        $this->assertEquals([6,4],$nextStates[0]->shape());
        $this->assertEquals($copyInputs->toArray(),$inputs->toArray());
        $this->assertEquals($copyStates[0]->toArray(),$initialStates[0]->toArray());

        //
        // backward
        //
        // 2 batch
        $dOutputs =
            $K->ones([6,5,4]);
        $dStates =
            [$K->ones([6,4])];

        $copydOutputs = $K->copy(
            $dOutputs);
        $copydStates = [$K->copy(
            $dStates[0])];
        //[$dInputs,$dPrevStates] = $layer->backward($dOutputs,$dStates);
        $dPrevStates = $layer->backward(array_merge([$dOutputs],$dStates));
        $dInputs = array_shift($dPrevStates);
        $this->assertCount(1,$dPrevStates);
        // 2 batch
        $this->assertEquals([6,5,3],$dInputs->shape());
        $this->assertCount(1,$dPrevStates);
        $this->assertEquals([6,4],$dPrevStates[0]->shape());
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

    public function testForwardAndBackwardWithReturnSeqquenceWithoutInitialStates()
    {
        $mo = new MatrixOperator();
        $K = $backend = $this->newBackend($mo);
        $fn = $backend;

        $layer = new SimpleRNN(
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
        $this->assertCount(1,$nextStates);
        $this->assertEquals([6,4],$nextStates[0]->shape());
        $this->assertEquals($copyInputs->toArray(),$inputs->toArray());
        //$this->assertEquals($copyStates[0]->toArray(),$initialStates[0]->toArray());

        //
        // backward
        //
        // 2 batch
        $dOutputs =
            $K->ones([6,5,4]);
        $dStates =
            [$K->ones([6,4])];

        $copydOutputs = $K->copy(
            $dOutputs);
        $copydStates = [$K->copy(
            $dStates[0])];
        //[$dInputs,$dPrevStates] = $layer->backward($dOutputs,$dStates);
        $dPrevStates = $layer->backward(array_merge([$dOutputs],$dStates));
        $dInputs = array_shift($dPrevStates);
        $this->assertCount(0,$dPrevStates);
        // 2 batch
        $this->assertEquals([6,5,3],$dInputs->shape());
        //$this->assertCount(1,$dPrevStates);
        //$this->assertEquals([6,4],$dPrevStates[0]->shape());
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

        $layer = new SimpleRNN(
            $backend,
            $units=4,
            [
                'input_shape'=>[3,5],
                'return_sequences'=>true,
                'return_state'=>true,
                'activation'=>null,
            ]);

        $kernel = $K->ones([5,4]);
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
        $inputs = $K->ones([2,3,5]);
        $states = [$K->ones([2,4])];
        [$outputs,$nextStates] = $layer->forward($inputs,$training=true, $states);
        //
        $this->assertEquals([
            [[10,10,10,10],[46,46,46,46],[190,190,190,190]],
            [[10,10,10,10],[46,46,46,46],[190,190,190,190]],
            ],$outputs->toArray());
        $this->assertEquals([
            [190,190,190,190],
            [190,190,190,190],
            ],$nextStates[0]->toArray());
        //
        // backward
        //
        // 2 batch
        $dOutputs =
            $K->ones([2,3,4]);
        $dStates =
            [$K->ones([2,4])];

        //[$dInputs,$dPrevStates] = $layer->backward($dOutputs,$dStates);
        $dPrevStates = $layer->backward(array_merge([$dOutputs],$dStates));
        $dInputs = array_shift($dPrevStates);
        $this->assertCount(1,$dPrevStates);
        // 2 batch
        $this->assertEquals([
            [[148,148,148,148,148],[36,36,36,36,36],[8,8,8,8,8],],
            [[148,148,148,148,148],[36,36,36,36,36],[8,8,8,8,8],],
            ],$dInputs->toArray());
        $this->assertEquals([
            [148,148,148,148],
            [148,148,148,148],
            ],$dPrevStates[0]->toArray());
        $this->assertEquals([
            [96,96,96,96],
            [96,96,96,96],
            [96,96,96,96],
            [96,96,96,96],
            [96,96,96,96],
            ],$grads[0]->toArray());
        $this->assertEquals([
            [438,438,438,438],
            [438,438,438,438],
            [438,438,438,438],
            [438,438,438,438],
            ],$grads[1]->toArray());
        $this->assertEquals(
            [96,96,96,96]
            ,$grads[2]->toArray());
    }

    public function testVerifyReturnSequences()
    {
        $mo = new MatrixOperator();
        $K = $backend = $this->newBackend($mo);
        $fn = $backend;

        $layer = new SimpleRNN(
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

    public function testVerifyGoBackwards()
    {
        $mo = new MatrixOperator();
        $K = $backend = $this->newBackend($mo);
        $fn = $backend;

        $layer = new SimpleRNN(
            $backend,
            $units=3,
            [
                'input_shape'=>[4,10],
                'return_sequences'=>true,
                'go_backwards'=>true,
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
