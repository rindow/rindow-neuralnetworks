<?php
namespace RindowTest\NeuralNetworks\Layer\SimpleRNNTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Layer\SimpleRNN;
use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Activation\Tanh;

class SimpleRNNTest extends TestCase
{
    public function newMatrixOperator()
    {
        return new MatrixOperator();
    }

    public function newNeuralNetworks($mo)
    {
        return new NeuralNetworks($mo);
    }

    public function verifyGradient($mo, $nn, $K, $g, $function, NDArray $x)
    {
        $f = function($x) use ($mo,$K,$function){
            $x = $K->array($x);
            $y = $function->forward($x);
            return $K->ndarray($y);
        };
        $grads = $mo->la()->numericalGradient(1e-3,$f,$K->ndarray($x));
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($function,$x) {
                $outputsVariable = $function->forward($x);
                return $outputsVariable;
            }
        );
        $dOutputs = $K->ones($outputsVariable->shape(),$outputsVariable->dtype());
        [$dInputs] = $outputsVariable->creator()->backward([$dOutputs]);
        return $mo->la()->isclose($grads[0],$K->ndarray($dInputs),1e-3);
    }

    public function testDefaultInitialize()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new SimpleRNN(
            $K,
            $units=4,
            input_shape:[5,3],
            );

        $inputs = [$g->Variable($K->zeros([1,5,3]))];
        $layer->build($inputs);
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

    public function testSetInputShape()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new SimpleRNN(
            $K,
            $units=4,
            );
        $inputs = [$g->Variable($K->zeros([1,5,3]))];
        $layer->build($inputs);

        //$this->assertEquals([3],$layer->inputShape());
        $this->assertEquals([4],$layer->outputShape());
    }

    public function testUnmatchSpecifiedInputShape()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new SimpleRNN(
            $K,
            $units=4,
            input_shape:[5,3],
            );

        $inputs = [$g->Variable($K->zeros([1,5,4]))];
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Input shape is inconsistent: defined as [5,3] but [5,4] given in SimpleRNN');
        $layer->build($inputs);
    }

    public function testSetInputShapeForSequential()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new SimpleRNN(
            $K,
            $units=4,
            );
        $inputs = $g->Variable($K->zeros([1,5,3]));
        $layer->build($inputs);

        //$this->assertEquals([3],$layer->inputShape());
        $this->assertEquals([4],$layer->outputShape());
    }

    public function testInitializeWithReturnSequence()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new SimpleRNN(
            $K,
            $units=4,
            input_shape:[5,3],
            return_sequences:true,
            return_state:true,
            );
        $inputs = [$g->Variable($K->zeros([1,5,3]))];
        $layer->build($inputs);

        //$this->assertEquals([3],$layer->inputShape());
        $this->assertEquals([5,4],$layer->outputShape());
    }

    public function testDefaultForwardAndBackward()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $fn = $K;

        $layer = new SimpleRNN(
            $K,
            $units=4,
            input_shape:[5,3],
            );

        //$layer->build();
        //$grads = $layer->getGrads();


        //
        // forward
        //
        //  2 batch
        $inputs = $K->ones([6,5,3]);
        $initialStates = [$K->ones([6,4])];
        $copyInputs = $K->copy($inputs);
        $copyStates = [$K->copy($initialStates[0])];
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$inputs,$initialStates) {
                $outputsVariable = $layer->forward($inputs,initialStates:$initialStates);
                return $outputsVariable;
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        $grads = $layer->getGrads();
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
        $dPrevStates = $outputsVariable->creator()->backward([$dOutputs]);
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
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $fn = $K;

        $layer = new SimpleRNN(
            $K,
            $units=4,
            input_shape:[5,3],
            return_sequences:true,
            return_state:true,
            );

        //$layer->build();
        //$grads = $layer->getGrads();


        //
        // forward
        //
        //  2 batch
        $inputs = $K->ones([6,5,3]);
        $initialStates = [$K->ones([6,4])];
        $copyInputs = $K->copy($inputs);
        $copyStates = [$K->copy($initialStates[0])];
        [$outputsVariable,$nextStatesVariable] = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$inputs,$initialStates) {
                [$outputsVariable,$nextStatesVariable] = $layer->forward($inputs,initialStates:$initialStates);
                return [$outputsVariable,$nextStatesVariable];
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        $nextStates = array_map(fn($x)=>$K->ndarray($x),$nextStatesVariable);
        $grads = $layer->getGrads();
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
        $dPrevStates = $outputsVariable->creator()->backward(array_merge([$dOutputs],$dStates));
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
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $fn = $K;

        $layer = new SimpleRNN(
            $K,
            $units=4,
            input_shape:[5,3],
            return_sequences:true,
            return_state:true,
            );

        //$layer->build();
        //$grads = $layer->getGrads();


        //
        // forward
        //
        //  2 batch
        $inputs = $K->ones([6,5,3]);
        $initialStates = null;
        $copyInputs = $K->copy($inputs);
        //$copyStates = [$mo->copy($initialStates[0])];
        [$outputsVariable,$nextStatesVariable] = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$inputs,$initialStates) {
                [$outputsVariable,$nextStatesVariable] = $layer->forward($inputs,initialStates:$initialStates);
                return [$outputsVariable,$nextStatesVariable];
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        $nextStates = array_map(fn($x)=>$K->ndarray($x),$nextStatesVariable);
        $grads = $layer->getGrads();
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
        $dPrevStates = $outputsVariable->creator()->backward(array_merge([$dOutputs],$dStates));
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
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $fn = $K;

        $layer = new SimpleRNN(
            $K,
            $units=4,
            input_shape:[3,5],
            return_sequences:true,
            return_state:true,
            activation:'linear',
            );


        //  2 batch
        $inputs = $K->ones([2,3,5]);
        $initialStates = [$K->ones([2,4])];
        // sampleWeights
        $kernel = $K->ones([5,4]);
        $recurrent = $K->ones([4,4]);
        $bias = $K->ones([4]);
        $layer->build(
            array_merge([$g->Variable($inputs)],array_map(fn($x)=>$g->Variable($x),$initialStates)),
            sampleWeights:[$kernel,$recurrent,$bias]
        );


        //
        // forward
        //
        [$outputsVariable,$nextStateVariables] = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$inputs,$initialStates) {
                [$outputsVariable,$nextStateVariables] = $layer->forward($inputs,initialStates:$initialStates);
                return [$outputsVariable,$nextStateVariables];
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        $nextStates = array_map(fn($x)=>$K->ndarray($x),$nextStateVariables);
        $this->assertNull($layer->getActivation());
        $grads = $layer->getGrads();
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
        $dPrevStates = $outputsVariable->creator()->backward(array_merge([$dOutputs],$dStates));
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
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $fn = $K;

        $layer = new SimpleRNN(
            $K,
            $units=3,
            input_shape:[4,10],
            return_sequences:true,
            #return_state:true,
            #activation:'linear',
            );
        //$layer->build();
        //$weights = $layer->getParams();

        $x = $K->array([
            [0,1,2,9],
        ],dtype:NDArray::int32);
        $x = $K->onehot($x->reshape([4]),$numClass=10)->reshape([1,4,10]);
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$x) {
                $outputsVariable = $layer->forward($x);
                return $outputsVariable;
            }
        );

        $this->assertTrue(
            $this->verifyGradient($mo,$nn,$K,$g,$layer,$x));
    }

    public function testVerifyGoBackwards()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $fn = $K;

        $layer = new SimpleRNN(
            $K,
            $units=3,
            input_shape:[4,10],
            return_sequences:true,
            go_backwards:true,
            #return_state:true,
            #activation:'linear',
            );
        //$layer->build();
        //$weights = $layer->getParams();

        $x = $K->array([
            [0,1,2,9],
        ],dtype:NDArray::int32);
        $x = $K->onehot($x->reshape([4]),$numClass=10)->reshape([1,4,10]);
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$x) {
                $outputsVariable = $layer->forward($x);
                return $outputsVariable;
            }
        );

        $this->assertTrue(
            $this->verifyGradient($mo,$nn,$K,$g,$layer,$x));
    }

    public function testClone()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $origLayer = new SimpleRNN(
            $K,
            $units=4,
            input_shape:[5,3],
            );

        $inputs = $g->Variable($K->zeros([1,5,3]));
        $inputs2 = $g->Variable($K->zeros([1,5,3]));
        $origLayer->build($inputs);
        $layer = clone $origLayer;
        $layer->build($inputs2);

        $origParams = $origLayer->getParams();
        $params = $layer->getParams();
        $this->assertCount(3,$params);
        foreach (array_map(null,$origParams,$params) as $data) {
            [$orig,$dest] = $data;
            $this->assertNotEquals(spl_object_id($orig),spl_object_id($dest));
        }
        $origParams = $origLayer->getGrads();
        $params = $layer->getGrads();
        $this->assertCount(3,$params);
        foreach (array_map(null,$origParams,$params) as $data) {
            [$orig,$dest] = $data;
            $this->assertNotEquals(spl_object_id($orig),spl_object_id($dest));
        }
    }
}
