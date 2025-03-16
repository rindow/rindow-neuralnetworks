<?php
namespace RindowTest\NeuralNetworks\Layer\EinsumDenseTest;

use InvalidArgumentException;
use PHPUnit\Framework\TestCase;
use PHPUnit\Framework\Attributes\DataProvider;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Layer\EinsumDense;

class EinsumDenseTest extends TestCase
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
        $dInputs = $outputsVariable->creator()->backward([$dOutputs]);

        return $mo->la()->isclose($grads[0],$K->ndarray($dInputs[0]),1e-3);
    }

    public static function providerDefaultInitialize()
    {
        return [
            "1d_end_weight" => [[
                "testcase_name" => "1d_end_weight",
                "equation" => "ab,b->a",
                "bias_axes" => null,
                "input_shape" => [2, 32/*:*/],
                "output_shape" => [],
                "expected_kernel_shape" => [32,],
                "expected_bias_shape" => null,
                "expected_output_shape" => [2,],
            ]],
            "2d_middle_weight" => [[
                "testcase_name" => "2d_middle_weight",
                "equation" => "ab,bc->ac",
                "bias_axes" => null,
                "input_shape" => [2, 32],
                "output_shape" => [64],
                "expected_kernel_shape" => [32, 64],
                "expected_bias_shape" => null,
                "expected_output_shape" => [2, 64],
            ]],
            "3d_bert" => [[
                "testcase_name" => "3d_bert",
                "equation" => "abc,cde->abde",
                "bias_axes" => null,
                "input_shape" => [2, 1, 2],
                "output_shape" => [1, 3, 4],
                "expected_kernel_shape" => [2, 3, 4],
                "expected_bias_shape" => null,
                "expected_output_shape" => [2, 1, 3, 4],
            ]],
            "3d_3_bias" => [[
                "testcase_name" => "3d_3_bias",
                "equation" => "abc,cde->abde",
                "bias_axes" => "e",
                "input_shape" => [2, 1, 2],
                "output_shape" => [1, 3, 4],
                "expected_kernel_shape" => [2, 3, 4],
                "expected_bias_shape" => [4,],
                "expected_output_shape" => [2, 1, 3, 4],
            ]],
            "3d_2_bias" => [[
                "testcase_name" => "3d_2_bias",
                "equation" => "abc,cde->abde",
                "bias_axes" => "d",
                "input_shape" => [2, 1, 2],
                "output_shape" => [1, 3, 4],
                "expected_kernel_shape" => [2, 3, 4],
                "expected_bias_shape" => [3, 1],
                "expected_output_shape" => [2, 1, 3, 4],
            ]],
            "3d_1_3_bias" => [[
                "testcase_name" => "3d_1_3_bias",
                "equation" => "abc,cde->abde",
                "bias_axes" => "be",
                "input_shape" => [2, 7, 2],
                "output_shape" => [7, 3, 4],
                "expected_kernel_shape" => [2, 3, 4],
                "expected_bias_shape" => [7, 1, 4],
                "expected_output_shape" => [2, 7, 3, 4],
            ]],
            "3d_bert_projection" => [[
                "testcase_name" => "3d_bert_projection",
                "equation" => "BFNH,NHD->BFD",
                "bias_axes" => null,
                "input_shape" => [2, 1, 2, 3],
                "output_shape" => [1, 4],
                "expected_kernel_shape" => [2, 3, 4],
                "expected_bias_shape" => null,
                "expected_output_shape" => [2, 1, 4],
            ]],
            "2d_bert" => [[
                "testcase_name" => "2d_bert",
                "equation" => "abc,cd->abd",
                "bias_axes" => null,
                "input_shape" => [2, 1, 2],
                "output_shape" => [1, 4],
                "expected_kernel_shape" => [2, 4],
                "expected_bias_shape" => null,
                "expected_output_shape" => [2, 1, 4],
            ]],
            "embedding_1d" => [[
                "testcase_name" => "embedding_1d",
                "equation" => "i,d->id",
                "bias_axes" => null,
                "input_shape" => [2,],
                "output_shape" => [2,],
                "expected_kernel_shape" => [2,],
                "expected_bias_shape" => null,
                "expected_output_shape" => [2, 2],
            ]],
            "xlnet_lm" => [[
                "testcase_name" => "xlnet_lm",
                "equation" => "ibd,nd->ibn",
                "bias_axes" => null,
                "input_shape" => [2, 2, 1],
                "output_shape" => [2, 2],
                "expected_kernel_shape" => [2, 1],
                "expected_bias_shape" => null,
                "expected_output_shape" => [2, 2, 2],
            ]],
            "2d_precast" => [[
                "testcase_name" => "2d_precast",
                "equation" => "...b,bc->...c",
                "bias_axes" => null,
                "input_shape" => [2, 32],
                "output_shape" => [64,],
                "expected_kernel_shape" => [32, 64],
                "expected_bias_shape" => null,
                "expected_output_shape" => [2, 64],
            ]],
            "2d_precast_elided_input_used_in_output" => [[
                "testcase_name" => "2d_precast_elided_input_used_in_output",
                "equation" => "...bc,bc->...b",
                "bias_axes" => null,
                "input_shape" => [2, 32, 64],
                "output_shape" => [32,],
                "expected_kernel_shape" => [32, 64],
                "expected_bias_shape" => null,
                "expected_output_shape" => [2, 32],
            ]],
            "2d_precast_multiple_elided_dims" => [[
                "testcase_name" => "2d_precast_multiple_elided_dims",
                "equation" => "...b,bc->...c",
                "bias_axes" => null,
                "input_shape" => [2, 3, 32],
                "output_shape" => [64,],
                "expected_kernel_shape" => [32, 64],
                "expected_bias_shape" => null,
                "expected_output_shape" => [2, 3, 64],
            ]],
            "3d_precast" => [[
                "testcase_name" => "3d_precast",
                "equation" => "...c,cde->...de",
                "bias_axes" => null,
                "input_shape" => [2, 1, 2],
                "output_shape" => [3, 4],
                "expected_kernel_shape" => [2, 3, 4],
                "expected_bias_shape" => null,
                "expected_output_shape" => [2, 1, 3, 4],
            ]],
            "3d_precast_3_bias" => [[
                "testcase_name" => "3d_precast_3_bias",
                "equation" => "...c,cde->...de",
                "bias_axes" => "e",
                "input_shape" => [2, 1, 2],
                "output_shape" => [3, 4],
                "expected_kernel_shape" => [2, 3, 4],
                "expected_bias_shape" => [4,],
                "expected_output_shape" => [2, 1, 3, 4],
            ]],
            "3d_precast_2_bias" => [[
                "testcase_name" => "3d_precast_2_bias",
                "equation" => "...c,cde->...de",
                "bias_axes" => "d",
                "input_shape" => [2, 1, 2],
                "output_shape" => [3, 4],
                "expected_kernel_shape" => [2, 3, 4],
                "expected_bias_shape" => [3, 1],
                "expected_output_shape" => [2, 1, 3, 4],
            ]],
            "3d_precast_2_3_bias" => [[
                "testcase_name" => "3d_precast_2_3_bias",
                "equation" => "...c,cde->...de",
                "bias_axes" => "de",
                "input_shape" => [2, 1, 2],
                "output_shape" => [3, 4],
                "expected_kernel_shape" => [2, 3, 4],
                "expected_bias_shape" => [3, 4],
                "expected_output_shape" => [2, 1, 3, 4],
            ]],
            "2d_postcast" => [[
                "testcase_name" => "2d_postcast",
                "equation" => "bc...,cd->bd...",
                "bias_axes" => null,
                "input_shape" => [2, 1, 2, 3],
                "output_shape" => [4,],
                "expected_kernel_shape" => [1, 4],
                "expected_bias_shape" => null,
                "expected_output_shape" => [2, 4, 2, 3],
            ]],
            "3d_postcast" => [[
                "testcase_name" => "3d_postcast",
                "equation" => "bc...,cde->bde...",
                "bias_axes" => null,
                "input_shape" => [2, 1, 2],
                "output_shape" => [3, 4],
                "expected_kernel_shape" => [1, 3, 4],
                "expected_bias_shape" => null,
                "expected_output_shape" => [2, 3, 4, 2],
            ]],
            "3d_postcast_1_bias" => [[
                "testcase_name" => "3d_postcast_1_bias",
                "equation" => "bc...,cde->bde...",
                "bias_axes" => "d",
                "input_shape" => [2, 1, 2],
                "output_shape" => [3, 4],
                "expected_kernel_shape" => [1, 3, 4],
                "expected_bias_shape" => [3, 1, 1],
                "expected_output_shape" => [2, 3, 4, 2],
            ]],
            "3d_postcast_2_bias" => [[
                "testcase_name" => "3d_postcast_2_bias",
                "equation" => "bc...,cde->bde...",
                "bias_axes" => "e",
                "input_shape" => [2, 1, 2],
                "output_shape" => [3, 4],
                "expected_kernel_shape" => [1, 3, 4],
                "expected_bias_shape" => [4, 1],
                "expected_output_shape" => [2, 3, 4, 2],
            ]],
            "3d_postcast_1_2_bias" => [[
                "testcase_name" => "3d_postcast_1_2_bias",
                "equation" => "bc...,cde->bde...",
                "bias_axes" => "de",
                "input_shape" => [2, 1, 2],
                "output_shape" => [3, 4],
                "expected_kernel_shape" => [1, 3, 4],
                "expected_bias_shape" => [3, 4, 1],
                "expected_output_shape" => [2, 3, 4, 2],
            ]],
        ];
    }

    #[DataProvider('providerDefaultInitialize')]
    public function testDefaultInitialize($params)
    {
        extract($params);
        $batch_size = array_shift($input_shape);

        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $layer = new EinsumDense(
            $K, $equation, $output_shape,
            input_shape: $input_shape,
            bias_axes: $bias_axes,
        );

        $layer->build();
        $params = $layer->getParams();
        if($bias_axes===null) {
            $this->assertCount(1,$params);
        } else {
            $this->assertCount(2,$params);
        }
        $this->assertEquals($expected_kernel_shape,$params[0]->shape());
        $this->assertNotEquals($mo->zeros($expected_kernel_shape)->toArray(),$params[0]->toArray());
        $this->assertEquals($expected_kernel_shape,$params[0]->shape());
        if($bias_axes!==null) {
            $this->assertEquals($expected_bias_shape,$params[1]->shape());
            $this->assertNotEquals($mo->zeros($expected_kernel_shape)->toArray(),$params[0]->toArray());
            $this->assertEquals($mo->zeros($expected_bias_shape)->toArray(),$params[1]->toArray());
        }

        $grads = $layer->getGrads();
        if($bias_axes===null) {
            $this->assertCount(1,$grads);
        } else {
            $this->assertCount(2,$grads);
        }
        $this->assertEquals($expected_kernel_shape,$grads[0]->shape());
        $this->assertEquals($mo->zeros($expected_kernel_shape)->toArray(),$grads[0]->toArray());
        if($bias_axes!==null) {
            $this->assertEquals($expected_bias_shape,$grads[1]->shape());
            $this->assertEquals($mo->zeros($expected_bias_shape)->toArray(),$grads[1]->toArray());
        }

        $this->assertEquals($expected_output_shape,array_merge([$batch_size],$layer->outputShape()));
        //$layer->unlink();
    }

    #[DataProvider('providerDefaultInitialize')]
    public function testSetInputShape($params)
    {
        extract($params);
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new EinsumDense(
            $K, $equation, $output_shape,
            bias_axes: $bias_axes,
        );
        $inputs = $g->Variable($K->zeros($input_shape));
        $layer->build($inputs);
        $params = $layer->getParams();
        if($bias_axes===null) {
            $this->assertCount(1,$params);
        } else {
            $this->assertCount(2,$params);
        }
        $this->assertEquals($expected_kernel_shape,$params[0]->shape());
        $this->assertNotEquals($mo->zeros($expected_kernel_shape)->toArray(),$params[0]->toArray());
        $this->assertEquals($expected_kernel_shape,$params[0]->shape());
        if($bias_axes!==null) {
            $this->assertEquals($expected_bias_shape,$params[1]->shape());
            $this->assertNotEquals($mo->zeros($expected_kernel_shape)->toArray(),$params[0]->toArray());
            $this->assertEquals($mo->zeros($expected_bias_shape)->toArray(),$params[1]->toArray());
        }
    }

    public function testUnmatchSpecifiedInputShape()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();

        $testcase_name = "3d_1_3_bias";
        $equation = "abc,cde->abde";
        $bias_axes = "be";
        $input_shape = [2, 7, 2];
        $output_shape = [7, 3, 4];
        $expected_kernel_shape = [2, 3, 4];
        $expected_bias_shape = [7, 1, 4];
        $expected_output_shape = [2, 7, 3, 4];

        $layer = new EinsumDense(
            $K, $equation, $output_shape,
            input_shape: $input_shape,
            bias_axes: $bias_axes,
        );

        $inputs = $g->Variable($K->zeros(array_merge([1],[2, 7, 5])));
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Input shape is inconsistent: defined as (2,7,2) but (2,7,5) given in EinsumDense');
        $layer->build($inputs);
    }

    public function testNormalForwardAndBackward()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $fn = $mo->la();

        $equation = "ab,bc->ac";
        $bias_axes = "c";
        $input_shape = [3];
        $output_shape = [2];

        $layer = new EinsumDense(
            $K, $equation, $output_shape,
            input_shape: $input_shape,
            bias_axes: $bias_axes,
            kernel_initializer:'ones',
            bias_initializer:'zeros',
        );

        // 3 input x 4 minibatch
        //$inputs = $K->ones(array_merge([4],$input_shape));
        $inputs = $K->array([
            [0.1, 0.2, 0.5],
            [0.1, 0.2, 0.5],
            [0.1, 0.2, 0.5],
            [0.1, 0.2, 0.5],
        ]);
        $full_output_shape = array_merge([4],$output_shape);

        $layer->build($g->Variable($inputs),
            //sampleWeights:[
            //    $K->array([[0.1, 0.2], [0.1, 0.1], [0.2, 0.2]]), // kernel
            //    $K->array([0.5, 0.1]),                           // bias
            //]
        );
        $params = $layer->getParams();
        $this->assertEquals(2,count($params));
        $this->assertEquals([3,2],$params[0]->shape());
        $this->assertEquals([2],$params[1]->shape());
        //$K->copy($K->array([[0.1, 0.2], [0.1, 0.1], [0.2, 0.2]]),$params[0]); // kernel
        //$K->copy($K->array([0.5, 0.1]),$params[1]); // bias

        //
        // forward
        //
        $copyInputs = $K->copy($inputs);
        $inputs = $K->array($inputs);
        $salt = $mo->la()->range(array_product($full_output_shape),dtype:NDArray::float32)
                ->reshape($full_output_shape);
        $salt = $g->Variable($salt);
        [$outputsVariable, $resultValiable] = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$layer,$inputs,$salt) {
                $outputsVariable = $layer->forward($inputs);
                $resultValiable = $g->mul($outputsVariable,$salt);
                return [$outputsVariable, $resultValiable];
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        $inputs = $K->ndarray($inputs);
        #echo "inputs: ".$mo->toString($inputs,indent:true)."\n";
        #echo "outputs: ".$mo->toString($outputs,indent:true)."\n";

        // 2 output x 4 batch
        $this->assertEquals([4,2],$outputs->shape());
        $this->assertTrue($fn->isclose($mo->array([
                [0.8, 0.8],
                [0.8, 0.8],
                [0.8, 0.8],
                [0.8, 0.8],
            ]),$outputs
        ));
        $this->assertEquals($copyInputs->toArray(),$inputs->toArray());

        //
        // backward
        //
        // 2 output x 4 batch
        $dResultValiable = $K->ones($resultValiable->shape());
        //$dOutputs = $salt;
        [$dOutputs,$dSalt] = $resultValiable->creator()->backward([$dResultValiable]);
        $this->assertEquals($outputsVariable->shape(),$dOutputs->shape());
        $this->assertEquals($resultValiable->shape(),$dSalt->shape());
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dOutputs),$K->ndarray($salt)
        ));
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dSalt),$outputs
        ));
        $copydOutputs = $K->copy($dOutputs);
        [$dInputs] = $outputsVariable->creator()->backward([$dOutputs]);
        $dInputs = $K->ndarray($dInputs);
        $dOutputs = $K->ndarray($dOutputs);
        #echo "dInputs: ".$mo->toString($dInputs,indent:true)."\n";
        // 3 input x 4 batch
        $this->assertEquals([4,3],$dInputs->shape());
        $this->assertTrue($fn->isclose($mo->array([
                [1.0, 1.0 ,1.0],
                [5.0, 5.0 ,5.0],
                [9.0, 9.0 ,9.0],
                [13.0, 13.0, 13.0]
            ]), $dInputs
        ));
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());
    }

    public function testNdInput()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $equation = "abc,cd->abd";
        $bias_axes = "d";
        $input_shape = [2,3];
        $output_shape = [2,4];

        $layer = new EinsumDense(
            $K, $equation, $output_shape,
            input_shape: $input_shape,
            bias_axes: $bias_axes,
        );

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

    public function testGradientWithActivation()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $fn = $mo->la();

        $equation = "ab,bc->ac";
        $bias_axes = "c";
        $input_shape = [3];
        $output_shape = [2];

        $layer = new EinsumDense(
            $K, $equation, $output_shape,
            input_shape: $input_shape,
            bias_axes: $bias_axes,
            activation:'tanh',
        );

        // 3 input x 4 minibatch
        $inputs = $K->ones([4,3]);

        $layer->build($g->Variable($inputs));

        //
        // forward
        //
        $copyInputs = $K->copy($inputs);
        $inputs = $K->array($inputs);
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$inputs) {
                $outputsVariable = $layer->forward($inputs);
                return $outputsVariable;
            }
        );

        $this->assertTrue(
            $this->verifyGradient($mo,$nn,$K,$g,$layer,$inputs));
    }
}
