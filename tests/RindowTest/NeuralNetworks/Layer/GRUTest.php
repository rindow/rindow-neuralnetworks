<?php
namespace RindowTest\NeuralNetworks\Layer\GRUTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Layer\GRU;
use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Activation\Tanh;
use Rindow\NeuralNetworks\Gradient\MaskedNDArray;

class GRUTest extends TestCase
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
#echo "\n";
#echo "grads=".$mo->toString($grads[0],'%5.3f',true)."\n\n";
#echo "dInputs=".$mo->toString($dInputs,'%5.3f',true)."\n\n";
#echo $mo->asum($mo->op($grads[0],'-',$dInputs))."\n";
        return $mo->la()->isclose($grads[0],$K->ndarray($dInputs),1e-1);#,1e-0,1e-1);
    }

    public function testDefaultInitialize()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new GRU(
            $K,
            $units=4,
            input_shape:[5,3],
            );
        $inputs = [$g->Variable($K->zeros([1,5,3]))];
        $layer->build($inputs);
        $params = $layer->getParams();
        $this->assertCount(3,$params);
        $this->assertEquals([3,4*3],$params[0]->shape());
        $this->assertEquals([4,4*3],$params[1]->shape());
        $this->assertEquals([2,4*3],$params[2]->shape());

        $grads = $layer->getGrads();
        $this->assertCount(3,$grads);
        $this->assertEquals([3,4*3],$grads[0]->shape());
        $this->assertEquals([4,4*3],$grads[1]->shape());
        $this->assertEquals([2,4*3],$grads[2]->shape());
        $this->assertNull(
            $layer->getActivation()
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
        $layer = new GRU(
            $K,
            $units=4,
            input_shape:[5,3],
            reset_after:false,
            );

        $inputs = [$g->Variable($K->zeros([1,5,3]))];
        $layer->build($inputs);
        $params = $layer->getParams();
        $this->assertCount(3,$params);
        $this->assertEquals([3,4*3],$params[0]->shape());
        $this->assertEquals([4*3,4],$params[1]->shape());
        $this->assertEquals([4*3],$params[2]->shape());

        $grads = $layer->getGrads();
        $this->assertCount(3,$grads);
        $this->assertEquals([3,4*3],$grads[0]->shape());
        $this->assertEquals([4*3,4],$grads[1]->shape());
        $this->assertEquals([4*3],$grads[2]->shape());
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
        $layer = new GRU(
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
        $layer = new GRU(
            $K,
            $units=4,
            input_shape:[5,3],
            );

        $inputs = [$g->Variable($K->zeros([1,5,4]))];
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Input shape is inconsistent: defined as (5,3) but (5,4) given in GRU');
        $layer->build($inputs);
    }

    public function testSetInputShapeForSequential()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new GRU(
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
        $layer = new GRU(
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

    public function testDefaultForwardAndBackwardWithInitialStates()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $fn = $K;

        $layer = new GRU(
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
        $copydStates = [
            $K->copy($dStates[0])];
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

    public function testDefaultForwardAndBackwardWithoutInitialStatesAnddStates()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $fn = $K;

        $layer = new GRU(
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
        $initialStates = null;
        $copyInputs = $K->copy($inputs);
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
        $dStates = null;

        $copydOutputs = $K->copy(
            $dOutputs);
        $dPrevStates = $outputsVariable->creator()->backward([$dOutputs]);
        $dInputs = array_shift($dPrevStates);
        $this->assertCount(0,$dPrevStates);
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
    }

    public function testForwardAndBackwardWithReturnSequence()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $fn = $K;

        $layer = new GRU(
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
        $initialStates = [
            $K->ones([6,4])];
        $copyInputs = $K->copy($inputs);
        $copyStates = [
            $K->copy($initialStates[0])];
        [$outputsVariable,$nextStateVariables] = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$inputs,$initialStates) {
                [$outputsVariable,$nextStateVariables] = $layer->forward($inputs,initialStates:$initialStates);
                return [$outputsVariable,$nextStateVariables];
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        $nextStates = array_map(fn($x)=>$K->ndarray($x),$nextStateVariables);
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
        $dStates = [
            $K->ones([6,4])];

        $copydOutputs = $K->copy(
            $dOutputs);
        $copydStates = [
            $K->copy($dStates[0])];
        //[$dInputs,$dPrevStates] = $layer->backward($dOutputs,$dStates);
        $dPrevStates = $outputsVariable->creator()->backward(array_merge([$dOutputs],$dStates));
        $dInputs = array_shift($dPrevStates);
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

    public function testForwardAndBackwardWithReturnSequenceWithoutInitialStates()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $fn = $K;

        $layer = new GRU(
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
        [$outputsVariable,$nextStateVariables] = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$inputs,$initialStates) {
                [$outputsVariable,$nextStateVariables] = $layer->forward($inputs,initialStates:$initialStates);
                return [$outputsVariable,$nextStateVariables];
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        $nextStates = array_map(fn($x)=>$K->ndarray($x),$nextStateVariables);
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
        $copydStates = [
            $K->copy($dStates[0])];
        //[$dInputs,$dPrevStates] = $layer->backward($dOutputs,$dStates);
        $dPrevStates = $outputsVariable->creator()->backward(array_merge([$dOutputs],$dStates));
        $dInputs = array_shift($dPrevStates);
        // 2 batch
        $this->assertEquals([6,5,3],$dInputs->shape());
        $this->assertCount(0,$dPrevStates);
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

        $layer = new GRU(
            $K,
            $units=4,
            input_shape:[3,5],
            return_sequences:true,
            return_state:true,
            activation:'linear',
            recurrent_activation:'linear',
            );


        //  2 batch
        $inputs = $K->ones([2,3,5]);
        $initialStates = [
            $K->ones([2,4])];
        // sample weights
        $kernel = $K->ones([5,4*3]);
        $recurrent = $K->ones([4,4*3]);
        $bias = $K->ones([2,4*3]);
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
        $grads = $layer->getGrads();
        $this->assertNull($layer->getActivation());
        //
        $this->assertEquals(
            [2,3,4],
            $outputs->shape());
        $this->assertEquals(
            [2,4],
            $nextStates[0]->shape());
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
        $this->assertEquals(
            [2,3,5],
            $dInputs->shape());
        $this->assertEquals(
            [2,4],
            $dPrevStates[0]->shape());
        $this->assertEquals(
            [5,12],
            $grads[0]->shape());
        $this->assertEquals(
            [4,12],
            $grads[1]->shape());
        $this->assertEquals(
            [2,12],
            $grads[2]->shape());
    }

    public function testVerifyReturnSequences()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $fn = $K;

        $layer = new GRU(
            $K,
            $units=3,
            input_shape:[4,10],
            return_sequences:true,
            #return_state:true,
            #activation:'linear',
            );
        $layer->build();
        $weights = $layer->getParams();

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
            $this->verifyGradient($mo,$nn,$K,$g,$layer,$x,1e-3));
    }

    public function testVerifyWithoutResetAfter()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $fn = $K;

        $layer = new GRU(
            $K,
            $units=3,
            input_shape:[4,10],
            return_sequences:true,
            #return_state:true,
            #activation:'linear',
            reset_after:false,
            );
        $layer->build();
        $weights = $layer->getParams();

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
        $origLayer = new GRU(
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
        foreach (array_map(null,$origParams,$params) as [$orig,$dest]) {
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

    public function testMaskedArrayForwardAndBackwardWithReturnSequence()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $fn = $K;

        $embedding = $nn->layers->Embedding(
            6, // inputDim
            3, // outputDim
            input_length:5,
            mask_zero:True,
        );
        $layer = new GRU(
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
        $seq = $K->array([
            [1,0,0,0,0],
            [1,2,0,0,0],
            [1,2,3,0,0],
            [1,2,3,4,0],
            [1,2,3,4,5],
            [1,2,3,4,5],
        ], dtype:NDArray::int32);
        $initialStates = [
            $K->ones([6,4])];
        $copyStates = [
            $K->copy($initialStates[0])];
        [$outputsVariable,$nextStateVariables] = $nn->with($tape=$g->GradientTape(),
            function() use ($embedding,$layer,$seq,$initialStates) {
                $inputs = $embedding($seq);
                [$outputsVariable,$nextStateVariables] = $layer->forward($inputs,initialStates:$initialStates);
                return [$outputsVariable,$nextStateVariables];
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        $nextStates = array_map(fn($x)=>$K->ndarray($x),$nextStateVariables);
        $grads = $layer->getGrads();
        //echo "outputs:".$mo->toString($outputs,'%8.5f',indent:true)."\n";
        //echo "next_h:".$mo->toString($nextStates[0],'%8.5f',indent:true)."\n";
        //
        $this->assertInstanceOf(MaskedNDArray::class,$outputsVariable->value());
        $this->assertEquals([6,5,4],$outputs->shape());
        $this->assertCount(1,$nextStates);
        $this->assertEquals([6,4],$nextStates[0]->shape());
        $this->assertEquals($copyStates[0]->toArray(),$initialStates[0]->toArray());

        //
        // backward
        //
        // 2 batch
        $dOutputs =
            $K->ones([6,5,4]);
        $dStates = [
            $K->ones([6,4])];

        $copydOutputs = $K->copy(
            $dOutputs);
        $copydStates = [
            $K->copy($dStates[0])];
        //[$dInputs,$dPrevStates] = $layer->backward($dOutputs,$dStates);
        $dPrevStates = $outputsVariable->creator()->backward(array_merge([$dOutputs],$dStates));
        $dInputs = array_shift($dPrevStates);
        //echo "dInputs:".$mo->toString($dInputs,'%8.5f',indent:true)."\n";
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

/*
    public function testDebug4()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $fn = $backend;

        $layer = new GRU(
            $backend,
            $units=10,
            input_shape:[4,10],
            return_sequences:true,
            #return_state:true,
            activation:'linear',
            #recurrent_activation:'linear',
            );
        $kernel = $mo->ones([10,30]);
        $recurrent_kernel = $mo->ones([10,30]);
        $bias = $mo->zeros([2,30]);
        $layer->build(null,
            sampleWeights:[
                $kernel,$recurrent_kernel,$bias,
            ]
        );
        $weights = $layer->getParams();
        //var_dump($weights[0]->shape());
        //var_dump($weights[1]->shape());
        //var_dump($weights[2]->shape());

        $x = $mo->array([
            [0,1,2,9],
        ],dtype:NDArray::int32);
        $x = $mo->la()->onehot($x->reshape([4]),$numClass=10)->reshape([1,4,10]);
        $outputs = $layer->forward($x);
        echo "outputs=".$mo->toString($outputs,'%5.3f',true);

        $ones = $mo->ones($outputs->shape());
        [$dInputs] = $layer->backward([$ones]);
        echo "dInputs=".$mo->toString($dInputs,'%5.3f',true);

        #$this->assertTrue(
        #    $this->verifyGradient($mo,$layer,$x));
    }

    public function testDebug1()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $fn = $backend;

        $layer = new GRU(
            $backend,
            $units=10,
                input_shape:[1,10],
                return_sequences:true,
                #return_state:true,
                activation:'linear',
                recurrent_activation:'linear',
                #reset_after:false,
            );
        $kernel = $mo->ones([10,30]);
        $recurrent_kernel = $mo->ones([10,30]);
        $bias = $mo->zeros([2,30]);
        #$recurrent_kernel = $mo->ones([30,10]);
        #$bias = $mo->zeros([30]);
        $layer->build(null,
            sampleWeights:[
                $kernel,$recurrent_kernel,$bias,
            ]
        );
        $weights = $layer->getParams();
        //var_dump($weights[0]->shape());
        //var_dump($weights[1]->shape());
        //var_dump($weights[2]->shape());

        $x = $mo->array([
            [1],
        ],dtype:NDArray::int32);
        $x = $mo->la()->onehot($x->reshape([1]),$numClass=10)->reshape([1,1,10]);
        $outputs = $layer->forward($x);
        echo "outputs=".$mo->toString($outputs,'%5.2f',true);

        $ones = $mo->ones($outputs->shape());
        [$dInputs] = $layer->backward([$ones]);
        #$this->assertTrue(
        #    $this->verifyGradient($mo,$layer,$x));
    }
*/
}
