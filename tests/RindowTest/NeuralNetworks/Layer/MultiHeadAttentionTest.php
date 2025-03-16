<?php
namespace RindowTest\NeuralNetworks\Layer\MultiHeadAttentionTest;

use PHPUnit\Framework\TestCase;
use PHPUnit\Framework\Attributes\DataProvider;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Matrix\Drivers\Service;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Layer\MultiHeadAttention;
use Rindow\NeuralNetworks\Gradient\MaskedNDArray;
use Rindow\NeuralNetworks\Gradient\Core\MaskedNDArray as MaskedNDArrayImpl;
use InvalidArgumentException;
use WeakMap;

class MultiHeadAttentionTest extends TestCase
{
    public function newMatrixOperator()
    {
        return new MatrixOperator();
    }

    public function newNeuralNetworks($mo)
    {
        return new NeuralNetworks($mo);
    }

    public function maskedValue(NDArray $value, NDArray $mask) : MaskedNDArray
    {
        return new MaskedNDArrayImpl($value,$mask);
    }

    public static function providerDefaultInitialize()
    {
        return [
            "input_key" => [[
                "num_heads"=>2,
                "key_dim"=>5,
                "value_dim"=>null,
                "use_bias"=>null,
                "dropout"=>null,
                "query_shape"=>[2, 8, 16],
                "value_shape"=>[2, 4, 16],
                "expected_output_shape"=>[2, 8, 16],
                "expected_num_trainable_weights"=>8,
                "expected_num_non_trainable_weights"=>0,
                "expected_num_seed_generators"=>0,
                "expected_num_losses"=>0,
                "supports_masking"=>true,
                "run_training_check"=>false,
            ]],
            "input_key_and_value" => [[
                "num_heads"=>2,
                "key_dim"=>5,
                "value_dim"=>6,
                "use_bias"=>false,
                "dropout"=>0.5,
                "query_shape"=>[2, 8, 16],
                "value_shape"=>[2, 4, 16],
                "expected_output_shape"=>[2, 8, 16],
                "expected_num_trainable_weights"=>4,
                "expected_num_non_trainable_weights"=>0,
                "expected_num_seed_generators"=>0,
                "expected_num_losses"=>0,
                "supports_masking"=>true,
                "run_training_check"=>false,
            ]],
        ];
    }

    #[DataProvider('providerDefaultInitialize')]
    public function testDefaultInitialize($params)
    {
        extract($params);
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();

        $batch_size = array_shift($query_shape);
        array_shift($value_shape);

        $layer = new MultiHeadAttention(
            $K,
            $num_heads, // num_heads
            $key_dim,   // key_dim
            value_dim:$value_dim,
            use_bias:$use_bias,
            input_shapes:[
                $query_shape, // query_shape
                $value_shape, // value_shape
            ],
        );
        $inputs = [
            $g->Variable($K->zeros(array_merge([$batch_size],$query_shape))),
            $g->Variable($K->zeros(array_merge([$batch_size],$value_shape))),
        ];

        $layer->build($inputs);
        $params = $layer->getParams();
        $this->assertCount($expected_num_trainable_weights,$params);
        //$this->assertEquals($expected_num_non_trainable_weights,$params[0]->shape());

        $grads = $layer->getGrads();
        $this->assertCount($expected_num_trainable_weights,$grads);

        array_shift($expected_output_shape);
        $this->assertEquals($expected_output_shape,$layer->outputShape());
    }

    #[DataProvider('providerDefaultInitialize')]
    public function testSetInputShape($params)
    {
        extract($params);
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $batch_size = array_shift($query_shape);
        array_shift($value_shape);
        $layer = new MultiHeadAttention(
            $K,
            $num_heads, // num_heads
            $key_dim,   // key_dim
            value_dim:$value_dim,
            use_bias:$use_bias,
        );
        $inputs = [
            $g->Variable($K->zeros(array_merge([$batch_size],$query_shape))),
            $g->Variable($K->zeros(array_merge([$batch_size],$value_shape))),
        ];
        $layer->build($inputs);
        array_shift($expected_output_shape);
        $this->assertEquals($expected_output_shape,$layer->outputShape());
    }

    public function testUnmatchSpecifiedInputShape()
    {
        $num_heads = 2;
        $key_dim = 5;
        $query_shape = [2, 8, 16];
        $value_shape = [2, 4, 16];

        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $batch_size = array_shift($query_shape);
        array_shift($value_shape);
        $layer = new MultiHeadAttention(
            $K,
            $num_heads, // num_heads
            $key_dim,   // key_dim
            input_shapes:[
                [8, 32], // query_shape
                [8, 16], // value_shape
            ],
        );
        $inputs = [
            $g->Variable($K->zeros(array_merge([$batch_size],$query_shape))),
            $g->Variable($K->zeros(array_merge([$batch_size],$value_shape))),
        ];
    
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Input shape is inconsistent: defined as ((8,32),(8,16)) but ((8,16),(4,16)) given in MultiHeadAttention');
        $layer->build($inputs);
    }

    public function testNormalForwardAndBackward()
    {
        $num_heads = 8;
        $key_dim = 4;
        #$full_query_shape = [2, 6, 16];
        #$full_value_shape = [2, 7, 16];
        $full_query_shape = [2, 6, 5];
        $full_value_shape = [2, 7, 5];
        #$full_query_shape = [2, 3, 6, 5];
        #$full_value_shape = [2, 3, 7, 5];
        $tmp = $full_query_shape;
        $tSeq = array_splice($tmp,1,-1);
        [$batches,$dim] = $tmp;
        $tmp = $full_value_shape;
        $sSeq = array_splice($tmp,1,-1);
        [$batches,$dim] = $tmp;

        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $la = $K->primaryLA();
        $layer = new MultiHeadAttention(
            $K,
            $num_heads, // num_heads
            $key_dim,   // key_dim
            kernel_initializer:'ones',
            bias_initializer:'zeros',
        );
        $salt_q = $mo->la()->range(array_product($full_query_shape),dtype:NDArray::float32)
                ->reshape($full_query_shape);
        $salt_v = $mo->la()->range(array_product($full_value_shape),dtype:NDArray::float32)
                ->reshape($full_value_shape);
        $salt_q = $K->array($salt_q);
        $salt_v = $K->array($salt_v);
        //$query = $g->Variable($la->randomNormal($full_query_shape,mean:0,scale:1));
        //$value = $g->Variable($la->randomNormal($full_value_shape,mean:0,scale:1));
        //echo "query: ".$mo->toString($dInputs[0],format:'%12.7e',indent:true)."\n";
        $query = $g->Variable($K->scale(1/array_product($full_query_shape),$K->increment($salt_q,1)));
        $value = $g->Variable($K->scale(1/array_product($full_value_shape),$K->increment($salt_v,1)));
        //echo 'query:'.$mo->toString($query,format:'%12.7f',indent:true)."\n";
        //echo 'value:'.$mo->toString($value,format:'%12.7f',indent:true)."\n";
        $inputs = [
            $query,
            $value,
        ];

        $layer->build($inputs,
        );

        //
        // forward
        //
        //  batch size 2
        //echo "query: ".$mo->toString($query,format:'%12.7f',indent:true)."\n";
        //////////////echo "key: ".$mo->toString($key,indent:true)."\n";
        //echo "value: ".$mo->toString($value,format:'%12.7f',indent:true)."\n";


        $salt = $g->Variable($salt_q);
        $copyInputs = [$K->copy($query),$K->copy($value)];
        [$outputsVariable,$scores,$resultValiable] = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$layer,$inputs,$salt) {
                [$outputsVariable,$scores] = $layer->forward(
                    $inputs,
                    training:true,
                    returnAttentionScores:true,
                );
                $resultValiable = $g->mul($outputsVariable,$salt);
                return [$outputsVariable,$scores,$resultValiable];
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        //echo 'query:'.$mo->toString($query,format:'%12.7f',indent:true)."\n";
        //echo 'value:'.$mo->toString($value,format:'%12.7f',indent:true)."\n";
        //echo 'outputs:'.$mo->toString($outputs,format:'%12.7f',indent:true)."\n";
        //echo 'scores:'.$mo->toString($scores,format:'%12.7e',indent:true)."\n";
        //echo 'kernel:'.$mo->toString($layer->getParams()[0],format:'%14.7f',indent:true)."\n";
        $this->assertEquals(array_merge([$batches, $num_heads], $tSeq, $sSeq),$scores->shape());
        $this->assertEquals(array_merge([$batches], $tSeq, [$dim]),$outputs->shape());
        $this->assertEquals($copyInputs[0]->toArray(),$inputs[0]->toArray());
        $this->assertEquals($copyInputs[1]->toArray(),$inputs[1]->toArray());
        //$this->assertTrue($mo->la()->isclose(
        //    $K->fill([2,8,6,7], 0.14285715),
        //    $K->ndarray($scores)));
        //$this->assertTrue($mo->la()->isclose(
        //    //$K->mul($salt,$K->fill($full_query_shape,512)),
        //    $K->fill($full_query_shape,512),
        //    $K->ndarray($outputs)
        //));
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($outputs),
            $mo->array([
               [[ 49.09698 ,  49.09698 ,  49.09698 ,  49.09698 ,  49.09698 ],
                [ 59.650528,  59.650528,  59.650528,  59.650528,  59.650528],
                [ 66.000854,  66.000854,  66.000854,  66.000854,  66.000854],
                [ 69.51905 ,  69.51905 ,  69.51905 ,  69.51905 ,  69.51905 ],
                [ 71.53566 ,  71.53566 ,  71.53566 ,  71.53566 ,  71.53566 ],
                [ 72.76805 ,  72.76805 ,  72.76805 ,  72.76805 ,  72.76805 ],],
              
               [[153.56422,  153.56422,  153.56422,  153.56422,  153.56422 ],
                [154.09987,  154.09987,  154.09987,  154.09987,  154.09987 ],
                [154.47055,  154.47055,  154.47055,  154.47055,  154.47055 ],
                [154.7322 ,  154.7322 ,  154.7322 ,  154.7322 ,  154.7322  ],
                [154.91945,  154.91945,  154.91945,  154.91945,  154.91945 ],
                [155.05478,  155.05478,  155.05478,  155.05478,  155.05478 ],],
           ])
        ));

        //echo "scores: ".$mo->toString($K->slice($scores,[0,0,0],[1,1,1]),format:'%12.7f',indent:true)."\n";
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($K->sum($scores,axis:1)),
            $mo->array([
              [[6.2805253e-01, 7.5084203e-01, 8.9763784e-01, 1.0731335e+00, 1.2829400e+00, 1.5337656e+00, 1.8336294e+00],
               [1.8050846e-01, 2.9060599e-01, 4.6785501e-01, 7.5321376e-01, 1.2126209e+00, 1.9522345e+00, 3.1429610e+00],
               [4.1690901e-02, 9.0386264e-02, 1.9595832e-01, 4.2483947e-01, 9.2105591e-01, 1.9968586e+00, 4.3292103e+00],
               [8.4980680e-03, 2.4810519e-02, 7.2435491e-02, 2.1147884e-01, 6.1742252e-01, 1.8025943e+00, 5.2627606e+00],
               [1.6152374e-03, 6.3504791e-03, 2.4967562e-02, 9.8162584e-02, 3.8593638e-01, 1.5173494e+00, 5.9656177e+00],
               [2.9460221e-04, 1.5597688e-03, 8.2581835e-03, 4.3722864e-02, 2.3149024e-01, 1.2256230e+00, 6.4890513e+00],],
             
              [[5.2358842e-05, 3.7330875e-04, 2.6616314e-03, 1.8976897e-02, 1.3530169e-01, 9.6467769e-01, 6.8779569e+00],
               [9.1480388e-06, 8.7833680e-05, 8.4332522e-04, 8.0970451e-03, 7.7742718e-02, 7.4643511e-01, 7.1667852e+00],
               [1.5798252e-06, 2.0426618e-05, 2.6410984e-04, 3.4148400e-03, 4.4152752e-02, 5.7088041e-01, 7.3812666e+00],
               [2.7061756e-07, 4.7119170e-06, 8.2042716e-05, 1.4284990e-03, 2.4872573e-02, 4.3307427e-01, 7.5405383e+00],
               [4.6088005e-08, 1.0806469e-06, 2.5338475e-05, 5.9412181e-04, 1.3930648e-02, 3.2663834e-01, 7.6588101e+00],
               [7.8165536e-09, 2.4681151e-07, 7.7932082e-06, 2.4607344e-04, 7.7698892e-03, 2.4533799e-01, 7.7466388e+00],],
            ])
        ));

        //
        // backward
        //
        // 
        $dResultValiable = $K->ones($resultValiable->shape());
        [$dOutputs,$dSalt] = $resultValiable->creator()->backward([$dResultValiable]);
        $this->assertEquals($outputsVariable->shape(),$dOutputs->shape());
        $this->assertEquals($resultValiable->shape(),$dSalt->shape());
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dOutputs),$K->ndarray($salt)
        ));
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dSalt),$K->ndarray($outputs)
        ));

        $copydOutputs = $K->copy($dOutputs);
        $dInputs = $outputsVariable->creator()->backward([$dOutputs]);
        // 2 batch
        $this->assertCount(2,$dInputs);
        //echo "dQuery: ".$mo->shapeToString($dInputs[0]->shape())."\n";
        //echo "dValue: ".$mo->shapeToString($dInputs[1]->shape())."\n";
        $this->assertEquals($full_query_shape,$dInputs[0]->shape());
        $this->assertEquals($full_value_shape,$dInputs[1]->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());

        //echo "dQuery: ".$mo->toString($dInputs[0],format:'%12.7e',indent:true)."\n";
        //echo "dValue: ".$mo->toString($dInputs[1],format:'%12.7e',indent:true)."\n";

        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dInputs[0]),
            $mo->array([
               [[302.03033,  302.03033,  302.03033,  302.03033,  302.03033 ],
                [699.38666,  699.38666,  699.38666,  699.38666,  699.38666 ],
                [670.80566,  670.80566,  670.80566,  670.80566,  670.80566 ],
                [530.9669 ,  530.9669 ,  530.9669 ,  530.9669 ,  530.9669  ],
                [407.75995,  407.75995,  407.75995,  407.75995,  407.75995 ],
                [315.91034,  315.91034,  315.91034,  315.91034,  315.91034 ],],
              
               [[247.77393 , 247.77393 , 247.77393 , 247.77393 , 247.77393 ],
                [195.96411 , 195.96411 , 195.96411 , 195.96411 , 195.96411 ],
                [155.72778 , 155.72778 , 155.72778 , 155.72778 , 155.72778 ],
                [123.993286, 123.993286, 123.993286, 123.993286, 123.993286],
                [ 98.746216,  98.746216,  98.746216,  98.746216,  98.746216],
                [ 78.53369 ,  78.53369 ,  78.53369 ,  78.53369 ,  78.53369 ],],
            ])
        ));
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dInputs[1]),
            $mo->array([
              [[-7.2089722e+01, -7.2089722e+01, -7.2089722e+01, -7.2089722e+01, -7.2089722e+01],
               [-1.1521368e+02, -1.1521368e+02, -1.1521368e+02, -1.1521368e+02, -1.1521368e+02],
               [-1.9455521e+02, -1.9455521e+02, -1.9455521e+02, -1.9455521e+02, -1.9455521e+02],
               [-3.2664609e+02, -3.2664609e+02, -3.2664609e+02, -3.2664609e+02, -3.2664609e+02],
               [-3.8889990e+02, -3.8889990e+02, -3.8889990e+02, -3.8889990e+02, -3.8889990e+02],
               [ 9.8908801e+02,  9.8908801e+02,  9.8908801e+02,  9.8908801e+02,  9.8908801e+02],
               [ 1.4028321e+04,  1.4028321e+04,  1.4028321e+04,  1.4028321e+04,  1.4028321e+04],],
             
              [[-4.5786795e-01, -4.5786795e-01, -4.5786795e-01, -4.5786795e-01, -4.5786795e-01],
               [-2.9618421e+00, -2.9618421e+00, -2.9618421e+00, -2.9618421e+00, -2.9618421e+00],
               [-1.9201010e+01, -1.9201010e+01, -1.9201010e+01, -1.9201010e+01, -1.9201010e+01],
               [-1.2483556e+02, -1.2483556e+02, -1.2483556e+02, -1.2483556e+02, -1.2483556e+02],
               [-7.9059326e+02, -7.9059326e+02, -7.9059326e+02, -7.9059326e+02, -7.9059326e+02],
               [-3.6179575e+03, -3.6179575e+03, -3.6179575e+03, -3.6179575e+03, -3.6179575e+03],
               [ 4.7275906e+04,  4.7275906e+04,  4.7275906e+04,  4.7275906e+04,  4.7275906e+04],],
            ])
        ));

    }

    public function testNormalWithKeyForwardAndBackward()
    {
        $num_heads = 8;
        $key_dim = 4;
        #$full_query_shape = [2, 6, 16];
        #$full_value_shape = [2, 7, 16];
        $full_query_shape = [2, 6, 5];
        $full_value_shape = [2, 7, 5];
        #$full_query_shape = [2, 3, 6, 5];
        #$full_value_shape = [2, 3, 7, 5];
        $tmp = $full_query_shape;
        $tSeq = array_splice($tmp,1,-1);
        [$batches,$dim] = $tmp;
        $tmp = $full_value_shape;
        $sSeq = array_splice($tmp,1,-1);
        [$batches,$dim] = $tmp;

        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $la = $K->primaryLA();
        $layer = new MultiHeadAttention(
            $K,
            $num_heads, // num_heads
            $key_dim,   // key_dim
            kernel_initializer:'ones',
            bias_initializer:'zeros',
        );
        $salt_q = $mo->la()->range(array_product($full_query_shape),dtype:NDArray::float32)
                ->reshape($full_query_shape);
        $salt_v = $mo->la()->range(array_product($full_value_shape),dtype:NDArray::float32)
                ->reshape($full_value_shape);
        $salt_q = $K->array($salt_q);
        $salt_v = $K->array($salt_v);
        //$query = $g->Variable($la->randomNormal($full_query_shape,mean:0,scale:1));
        //$value = $g->Variable($la->randomNormal($full_value_shape,mean:0,scale:1));
        //echo "query: ".$mo->toString($dInputs[0],format:'%12.7e',indent:true)."\n";
        $query = $g->Variable($K->scale(1/array_product($full_query_shape),$K->increment($salt_q,1)));
        $value = $g->Variable($K->scale(1/array_product($full_value_shape),$K->increment($salt_v,1)));
        $key = $g->Variable($K->scale(1/array_product($full_value_shape),$K->increment($salt_v,1)));
        //echo 'query:'.$mo->toString($query,format:'%12.7f',indent:true)."\n";
        //echo 'value:'.$mo->toString($value,format:'%12.7f',indent:true)."\n";
        $inputs = [
            $query,
            $value,
            $key,
        ];

        $layer->build($inputs,
        );

        //
        // forward
        //
        //  batch size 2
        //echo "query: ".$mo->toString($query,format:'%12.7f',indent:true)."\n";
        //////////////echo "key: ".$mo->toString($key,indent:true)."\n";
        //echo "value: ".$mo->toString($value,format:'%12.7f',indent:true)."\n";


        $salt = $g->Variable($salt_q);
        $copyInputs = [$K->copy($query),$K->copy($value)];
        [$outputsVariable,$scores,$resultValiable] = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$layer,$inputs,$salt) {
                [$outputsVariable,$scores] = $layer->forward(
                    $inputs,
                    training:true,
                    returnAttentionScores:true,
                );
                $resultValiable = $g->mul($outputsVariable,$salt);
                return [$outputsVariable,$scores,$resultValiable];
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        //echo 'query:'.$mo->toString($query,format:'%12.7f',indent:true)."\n";
        //echo 'value:'.$mo->toString($value,format:'%12.7f',indent:true)."\n";
        //echo 'outputs:'.$mo->toString($outputs,format:'%12.7f',indent:true)."\n";
        //echo 'scores:'.$mo->toString($scores,format:'%12.7e',indent:true)."\n";
        //echo 'kernel:'.$mo->toString($layer->getParams()[0],format:'%14.7f',indent:true)."\n";
        $this->assertEquals(array_merge([$batches, $num_heads], $tSeq, $sSeq),$scores->shape());
        $this->assertEquals(array_merge([$batches], $tSeq, [$dim]),$outputs->shape());
        $this->assertEquals($copyInputs[0]->toArray(),$inputs[0]->toArray());
        $this->assertEquals($copyInputs[1]->toArray(),$inputs[1]->toArray());
        //$this->assertTrue($mo->la()->isclose(
        //    $K->fill([2,8,6,7], 0.14285715),
        //    $K->ndarray($scores)));
        //$this->assertTrue($mo->la()->isclose(
        //    //$K->mul($salt,$K->fill($full_query_shape,512)),
        //    $K->fill($full_query_shape,512),
        //    $K->ndarray($outputs)
        //));
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($outputs),
            $mo->array([
               [[ 49.09698 ,  49.09698 ,  49.09698 ,  49.09698 ,  49.09698 ],
                [ 59.650528,  59.650528,  59.650528,  59.650528,  59.650528],
                [ 66.000854,  66.000854,  66.000854,  66.000854,  66.000854],
                [ 69.51905 ,  69.51905 ,  69.51905 ,  69.51905 ,  69.51905 ],
                [ 71.53566 ,  71.53566 ,  71.53566 ,  71.53566 ,  71.53566 ],
                [ 72.76805 ,  72.76805 ,  72.76805 ,  72.76805 ,  72.76805 ],],
              
               [[153.56422,  153.56422,  153.56422,  153.56422,  153.56422 ],
                [154.09987,  154.09987,  154.09987,  154.09987,  154.09987 ],
                [154.47055,  154.47055,  154.47055,  154.47055,  154.47055 ],
                [154.7322 ,  154.7322 ,  154.7322 ,  154.7322 ,  154.7322  ],
                [154.91945,  154.91945,  154.91945,  154.91945,  154.91945 ],
                [155.05478,  155.05478,  155.05478,  155.05478,  155.05478 ],],
           ])
        ));

        //echo "scores: ".$mo->toString($K->slice($scores,[0,0,0],[1,1,1]),format:'%12.7f',indent:true)."\n";
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($K->sum($scores,axis:1)),
            $mo->array([
              [[6.2805253e-01, 7.5084203e-01, 8.9763784e-01, 1.0731335e+00, 1.2829400e+00, 1.5337656e+00, 1.8336294e+00],
               [1.8050846e-01, 2.9060599e-01, 4.6785501e-01, 7.5321376e-01, 1.2126209e+00, 1.9522345e+00, 3.1429610e+00],
               [4.1690901e-02, 9.0386264e-02, 1.9595832e-01, 4.2483947e-01, 9.2105591e-01, 1.9968586e+00, 4.3292103e+00],
               [8.4980680e-03, 2.4810519e-02, 7.2435491e-02, 2.1147884e-01, 6.1742252e-01, 1.8025943e+00, 5.2627606e+00],
               [1.6152374e-03, 6.3504791e-03, 2.4967562e-02, 9.8162584e-02, 3.8593638e-01, 1.5173494e+00, 5.9656177e+00],
               [2.9460221e-04, 1.5597688e-03, 8.2581835e-03, 4.3722864e-02, 2.3149024e-01, 1.2256230e+00, 6.4890513e+00],],
             
              [[5.2358842e-05, 3.7330875e-04, 2.6616314e-03, 1.8976897e-02, 1.3530169e-01, 9.6467769e-01, 6.8779569e+00],
               [9.1480388e-06, 8.7833680e-05, 8.4332522e-04, 8.0970451e-03, 7.7742718e-02, 7.4643511e-01, 7.1667852e+00],
               [1.5798252e-06, 2.0426618e-05, 2.6410984e-04, 3.4148400e-03, 4.4152752e-02, 5.7088041e-01, 7.3812666e+00],
               [2.7061756e-07, 4.7119170e-06, 8.2042716e-05, 1.4284990e-03, 2.4872573e-02, 4.3307427e-01, 7.5405383e+00],
               [4.6088005e-08, 1.0806469e-06, 2.5338475e-05, 5.9412181e-04, 1.3930648e-02, 3.2663834e-01, 7.6588101e+00],
               [7.8165536e-09, 2.4681151e-07, 7.7932082e-06, 2.4607344e-04, 7.7698892e-03, 2.4533799e-01, 7.7466388e+00],],
            ])
        ));

        //
        // backward
        //
        // 
        $dResultValiable = $K->ones($resultValiable->shape());
        [$dOutputs,$dSalt] = $resultValiable->creator()->backward([$dResultValiable]);
        $this->assertEquals($outputsVariable->shape(),$dOutputs->shape());
        $this->assertEquals($resultValiable->shape(),$dSalt->shape());
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dOutputs),$K->ndarray($salt)
        ));
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dSalt),$K->ndarray($outputs)
        ));

        $copydOutputs = $K->copy($dOutputs);
        $dInputs = $outputsVariable->creator()->backward([$dOutputs]);
        // 2 batch
        $this->assertCount(3,$dInputs);
        //echo "dQuery: ".$mo->shapeToString($dInputs[0]->shape())."\n";
        //echo "dValue: ".$mo->shapeToString($dInputs[1]->shape())."\n";
        $this->assertEquals($full_query_shape,$dInputs[0]->shape());
        $this->assertEquals($full_value_shape,$dInputs[1]->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());

        //echo "dQuery: ".$mo->toString($dInputs[0],format:'%12.7e',indent:true)."\n";
        //echo "dValue: ".$mo->toString($dInputs[1],format:'%12.7e',indent:true)."\n";

        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dInputs[0]),
            $mo->array([
               [[302.03033,  302.03033,  302.03033,  302.03033,  302.03033 ],
                [699.38666,  699.38666,  699.38666,  699.38666,  699.38666 ],
                [670.80566,  670.80566,  670.80566,  670.80566,  670.80566 ],
                [530.9669 ,  530.9669 ,  530.9669 ,  530.9669 ,  530.9669  ],
                [407.75995,  407.75995,  407.75995,  407.75995,  407.75995 ],
                [315.91034,  315.91034,  315.91034,  315.91034,  315.91034 ],],
              
               [[247.77393 , 247.77393 , 247.77393 , 247.77393 , 247.77393 ],
                [195.96411 , 195.96411 , 195.96411 , 195.96411 , 195.96411 ],
                [155.72778 , 155.72778 , 155.72778 , 155.72778 , 155.72778 ],
                [123.993286, 123.993286, 123.993286, 123.993286, 123.993286],
                [ 98.746216,  98.746216,  98.746216,  98.746216,  98.746216],
                [ 78.53369 ,  78.53369 ,  78.53369 ,  78.53369 ,  78.53369 ],],
            ])
        ));
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dInputs[1]),
            $mo->array([
              [[6.41582184e+01, 6.41582184e+01, 6.41582184e+01, 6.41582184e+01, 6.41582184e+01],
               [1.04483284e+02, 1.04483284e+02, 1.04483284e+02, 1.04483284e+02, 1.04483284e+02],
               [1.88508392e+02, 1.88508392e+02, 1.88508392e+02, 1.88508392e+02, 1.88508392e+02],
               [3.89041473e+02, 3.89041473e+02, 3.89041473e+02, 3.89041473e+02, 3.89041473e+02],
               [9.46878174e+02, 9.46878174e+02, 9.46878174e+02, 9.46878174e+02, 9.46878174e+02],
               [2.75626172e+03, 2.75626172e+03, 2.75626172e+03, 2.75626172e+03, 2.75626172e+03],
               [9.47066992e+03, 9.47066992e+03, 9.47066992e+03, 9.47066992e+03, 9.47066992e+03],],
             
              [[4.19174954e-02, 4.19174954e-02, 4.19174954e-02, 4.19174954e-02, 4.19174954e-02],
               [3.26907337e-01, 3.26907337e-01, 3.26907337e-01, 3.26907337e-01, 3.26907337e-01],
               [2.66171312e+00, 2.66171312e+00, 2.66171312e+00, 2.66171312e+00, 2.66171312e+00],
               [2.32466946e+01, 2.32466946e+01, 2.32466946e+01, 2.32466946e+01, 2.32466946e+01],
               [2.27936752e+02, 2.27936752e+02, 2.27936752e+02, 2.27936752e+02, 2.27936752e+02],
               [2.67577393e+03, 2.67577393e+03, 2.67577393e+03, 2.67577393e+03, 2.67577393e+03],
               [3.97900078e+04, 3.97900078e+04, 3.97900078e+04, 3.97900078e+04, 3.97900078e+04],],
            ])
        ));

        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dInputs[2]),
            $mo->array([
              [[-1.3624794e+02, -1.3624794e+02, -1.3624794e+02, -1.3624794e+02, -1.3624794e+02],
               [-2.1969696e+02, -2.1969696e+02, -2.1969696e+02, -2.1969696e+02, -2.1969696e+02],
               [-3.8306360e+02, -3.8306360e+02, -3.8306360e+02, -3.8306360e+02, -3.8306360e+02],
               [-7.1568756e+02, -7.1568756e+02, -7.1568756e+02, -7.1568756e+02, -7.1568756e+02],
               [-1.3357781e+03, -1.3357781e+03, -1.3357781e+03, -1.3357781e+03, -1.3357781e+03],
               [-1.7671737e+03, -1.7671737e+03, -1.7671737e+03, -1.7671737e+03, -1.7671737e+03],
               [ 4.5576514e+03,  4.5576514e+03,  4.5576514e+03,  4.5576514e+03,  4.5576514e+03],],
             
              [[-4.9978545e-01, -4.9978545e-01, -4.9978545e-01, -4.9978545e-01, -4.9978545e-01],
               [-3.2887495e+00, -3.2887495e+00, -3.2887495e+00, -3.2887495e+00, -3.2887495e+00],
               [-2.1862722e+01, -2.1862722e+01, -2.1862722e+01, -2.1862722e+01, -2.1862722e+01],
               [-1.4808226e+02, -1.4808226e+02, -1.4808226e+02, -1.4808226e+02, -1.4808226e+02],
               [-1.0185300e+03, -1.0185300e+03, -1.0185300e+03, -1.0185300e+03, -1.0185300e+03],
               [-6.2937314e+03, -6.2937314e+03, -6.2937314e+03, -6.2937314e+03, -6.2937314e+03],
               [ 7.4858965e+03,  7.4858965e+03,  7.4858965e+03,  7.4858965e+03,  7.4858965e+03],],
            ]),
        ));

    }

    public function testNormal4DForwardAndBackward()
    {
        $num_heads = 8;
        $key_dim = 4;
        #$full_query_shape = [2, 6, 16];
        #$full_value_shape = [2, 7, 16];
        #$full_query_shape = [2, 3, 6];
        #$full_value_shape = [2, 5, 6];
        $full_query_shape = [2, 3, 6, 5];
        $full_value_shape = [2, 3, 7, 5];
        $tmp = $full_query_shape;
        $tSeq = array_splice($tmp,1,-1);
        [$batches,$dim] = $tmp;
        $tmp = $full_value_shape;
        $sSeq = array_splice($tmp,1,-1);
        [$batches,$dim] = $tmp;

        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $la = $K->primaryLA();
        $layer = new MultiHeadAttention(
            $K,
            $num_heads, // num_heads
            $key_dim,   // key_dim
            kernel_initializer:'ones',
            bias_initializer:'zeros',
        );
        $salt_q = $mo->la()->range(array_product($full_query_shape),dtype:NDArray::float32)
                ->reshape($full_query_shape);
        $salt_v = $mo->la()->range(array_product($full_value_shape),dtype:NDArray::float32)
                ->reshape($full_value_shape);
        $salt_q = $K->array($salt_q);
        $salt_v = $K->array($salt_v);
        //$query = $g->Variable($la->randomNormal($full_query_shape,mean:0,scale:1));
        //$value = $g->Variable($la->randomNormal($full_value_shape,mean:0,scale:1));
        //echo "query: ".$mo->toString($dInputs[0],format:'%12.7e',indent:true)."\n";
        $query = $g->Variable($K->scale(1/array_product($full_query_shape),$K->increment($salt_q,1)));
        $value = $g->Variable($K->scale(1/array_product($full_value_shape),$K->increment($salt_v,1)));
        //$query = $g->Variable($K->increment(
        //        $K->scale(0.5, $K->array($mo->la()->range(array_product($full_query_shape),dtype:NDArray::float32)
        //        ->reshape($full_query_shape))),
        //    1,
        //));
        //$value = $g->Variable($K->increment(
        //        $K->scale(0.2, $K->array($mo->la()->range(array_product($full_value_shape),dtype:NDArray::float32)
        //        ->reshape($full_value_shape))),
        //    1,
        //));
        //echo 'query:'.$mo->toString($query,format:'%12.7f',indent:true)."\n";
        //echo 'value:'.$mo->toString($value,format:'%12.7f',indent:true)."\n";
        $inputs = [
            $query,
            $value,
        ];

        $layer->build($inputs,
            //sampleWeights:[
            //    $K->ones([$num_heads,  $key_dim, $dim]),    // query_dense/kernel
            //    $K->zeros([$num_heads, $key_dim]),          // query_dense/bias
            //    $K->ones([$num_heads,  $key_dim, $dim]),    // key_dense/kernel
            //    $K->zeros([$num_heads, $key_dim]),          // key_dense/bias
            //    $K->ones([$num_heads,  $key_dim, $dim]),    // value_dense/kernel
            //    $K->zeros([$num_heads, $key_dim]),          // value_dense/bias
            //    $K->ones([$num_heads,  $key_dim, $dim]),    // output_dense/kernel
            //    $K->zeros([$dim]),                          // output_dense/bias
            //]
        );

        //
        // forward
        //
        //  batch size 2
        //echo "query: ".$mo->toString($query,format:'%12.7f',indent:true)."\n";
        //////////////echo "key: ".$mo->toString($key,indent:true)."\n";
        //echo "value: ".$mo->toString($value,format:'%12.7f',indent:true)."\n";


        $salt = $g->Variable($salt_q);
        $copyInputs = [$K->copy($query),$K->copy($value)];
        [$outputsVariable,$scores,$resultValiable] = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$layer,$inputs,$salt) {
                [$outputsVariable,$scores] = $layer->forward(
                    $inputs,
                    training:true,
                    returnAttentionScores:true,
                );
                $resultValiable = $g->mul($outputsVariable,$salt);
                return [$outputsVariable,$scores,$resultValiable];
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        //echo 'query:'.$mo->toString($query,format:'%12.7f',indent:true)."\n";
        //echo 'value:'.$mo->toString($value,format:'%12.7f',indent:true)."\n";
        //echo 'outputs:'.$mo->toString($outputs,format:'%12.7f',indent:true)."\n";
        //echo 'scores:'.$mo->toString($scores,format:'%12.7e',indent:true)."\n";
        //echo 'kernel:'.$mo->toString($layer->getParams()[0],format:'%14.7f',indent:true)."\n";
        $this->assertEquals(array_merge([$batches, $num_heads], $tSeq, $sSeq),$scores->shape());
        $this->assertEquals(array_merge([$batches], $tSeq, [$dim]),$outputs->shape());
        $this->assertEquals($copyInputs[0]->toArray(),$inputs[0]->toArray());
        $this->assertEquals($copyInputs[1]->toArray(),$inputs[1]->toArray());
        //$this->assertTrue($mo->la()->isclose(
        //    $K->fill([2,8,6,7], 0.14285715),
        //    $K->ndarray($scores)));
        //$this->assertTrue($mo->la()->isclose(
        //    //$K->mul($salt,$K->fill($full_query_shape,512)),
        //    $K->fill($full_query_shape,512),
        //    $K->ndarray($outputs)
        //));
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($outputs),
            $mo->array([
              [[[ 43.14443 ,  43.14443 ,  43.14443 ,  43.14443 ,  43.14443 ],
                [ 47.62348 ,  47.62348 ,  47.62348 ,  47.62348 ,  47.62348 ],
                [ 51.78363 ,  51.78363 ,  51.78363 ,  51.78363 ,  51.78363 ],
                [ 55.49719 ,  55.49719 ,  55.49719 ,  55.49719 ,  55.49719 ],
                [ 58.708492,  58.708492,  58.708492,  58.708492,  58.708492],
                [ 61.42242 ,  61.42242 ,  61.42242 ,  61.42242 ,  61.42242 ],],
             
               [[ 63.683285,  63.683285,  63.683285,  63.683285,  63.683285],
                [ 65.55373 ,  65.55373 ,  65.55373 ,  65.55373 ,  65.55373 ],
                [ 67.099846,  67.099846,  67.099846,  67.099846,  67.099846],
                [ 68.38227 ,  68.38227 ,  68.38227 ,  68.38227 ,  68.38227 ],
                [ 69.45283 ,  69.45283 ,  69.45283 ,  69.45283 ,  69.45283 ],
                [ 70.35382 ,  70.35382 ,  70.35382 ,  70.35382 ,  70.35382 ],],
             
               [[ 71.11888 ,  71.11888 ,  71.11888 ,  71.11888 ,  71.11888 ],
                [ 71.7744  ,  71.7744  ,  71.7744  ,  71.7744  ,  71.7744  ],
                [ 72.34105 ,  72.34105 ,  72.34105 ,  72.34105 ,  72.34105 ],
                [ 72.834854,  72.834854,  72.834854,  72.834854,  72.834854],
                [ 73.26857 ,  73.26857 ,  73.26857 ,  73.26857 ,  73.26857 ],
                [ 73.652176,  73.652176,  73.652176,  73.652176,  73.652176],]],
             
             
              [[[153.99342,  153.99342,  153.99342,  153.99342,  153.99342 ],
                [154.29929,  154.29929,  154.29929,  154.29929,  154.29929 ],
                [154.5742 ,  154.5742 ,  154.5742 ,  154.5742 ,  154.5742  ],
                [154.82278,  154.82278,  154.82278,  154.82278,  154.82278 ],
                [155.04842,  155.04842,  155.04842,  155.04842,  155.04842 ],
                [155.2543 ,  155.2543 ,  155.2543 ,  155.2543 ,  155.2543  ],],
             
               [[155.4424 ,  155.4424 ,  155.4424 ,  155.4424 ,  155.4424  ],
                [155.61523,  155.61523,  155.61523,  155.61523,  155.61523 ],
                [155.77469,  155.77469,  155.77469,  155.77469,  155.77469 ],
                [155.92142,  155.92142,  155.92142,  155.92142,  155.92142 ],
                [156.05716,  156.05716,  156.05716,  156.05716,  156.05716 ],
                [156.18294,  156.18294,  156.18294,  156.18294,  156.18294 ],],
             
               [[156.3003 ,  156.3003 ,  156.3003 ,  156.3003 ,  156.3003  ],
                [156.40952,  156.40952,  156.40952,  156.40952,  156.40952 ],
                [156.5112 ,  156.5112 ,  156.5112 ,  156.5112 ,  156.5112  ],
                [156.60686,  156.60686,  156.60686,  156.60686,  156.60686 ],
                [156.69571,  156.69571,  156.69571,  156.69571,  156.69571 ],
                [156.77946,  156.77946,  156.77946,  156.77946,  156.77946 ],]],

           ])
        ));

        //echo "scores: ".$mo->toString($K->slice($scores,[0,0,0],[1,1,1]),format:'%12.7f',indent:true)."\n";
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($K->slice($scores,[0,0,0],[1,1,1])),
            $mo->array([[[[
              [[0.0387687 , 0.03954561, 0.04033808, 0.04114642, 0.04197097, 0.04281205, 0.04366998],
               [0.0445451 , 0.04543776, 0.0463483 , 0.04727709, 0.04822451, 0.04919089, 0.05017664],
               [0.05118216, 0.05220782, 0.05325403, 0.05432122, 0.05540978, 0.05652016, 0.05765278],],
             
              [[0.02666436, 0.02811315, 0.02964068, 0.0312512 , 0.03294923, 0.03473952, 0.03662709],
               [0.0386172 , 0.04071546, 0.04292773, 0.04526021, 0.04771941, 0.05031224, 0.05304594],
               [0.05592817, 0.05896702, 0.06217099, 0.06554904, 0.06911063, 0.07286575, 0.0768249 ],],
             
              [[0.01766222, 0.019248  , 0.02097615, 0.02285945, 0.02491185, 0.02714851, 0.02958599],
               [0.03224233, 0.03513715, 0.03829188, 0.04172985, 0.04547648, 0.04955952, 0.05400913],
               [0.05885824, 0.06414273, 0.06990168, 0.07617766, 0.08301716, 0.09047072, 0.09859344],],
             
              [[0.01130549, 0.01273477, 0.01434473, 0.01615825, 0.01820103, 0.02050208, 0.02309402],
               [0.02601364, 0.02930238, 0.03300689, 0.03717974, 0.04188014, 0.04717477, 0.05313878],
               [0.05985678, 0.06742408, 0.07594807, 0.08554968, 0.09636521, 0.10854804, 0.12227104],],
             
              [[0.0070222 , 0.00817592, 0.00951918, 0.01108314, 0.01290406, 0.01502414, 0.01749255],
               [0.0203665 , 0.02371263, 0.02760851, 0.03214447, 0.03742568, 0.04357456, 0.05073366],
               [0.059069  , 0.06877378, 0.08007302, 0.09322865, 0.10854576, 0.12637939, 0.14714293],],
             
              [[0.00425106, 0.0051159 , 0.00615669, 0.00740921, 0.00891655, 0.01073055, 0.01291359],
               [0.01554074, 0.01870237, 0.02250721, 0.0270861 , 0.03259653, 0.03922801, 0.04720861],
               [0.05681279, 0.06837086, 0.08228031, 0.0990195 , 0.11916418, 0.14340709, 0.17258199],],               
            ]]]])
        ));

        //echo "scores: ".$mo->toString($K->slice($scores,[-1,-1,-1],[1,1,1]),format:'%12.7e',indent:true)."\n";
        //echo "scores: ".$mo->shapeToString($K->slice($scores,[-1,-1,-1],[1,1,1])->shape())."\n";
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($K->slice($scores,[-1,-1,-1],[1,1,1])),
            $mo->array([[[[
              [[1.03392339e-09, 2.84414514e-09, 7.82378340e-09, 2.15220020e-08, 5.92033196e-08, 1.62858456e-07, 4.47997223e-07],
               [1.23236555e-06, 3.39005123e-06, 9.32544754e-06, 2.56526982e-05, 7.05664206e-05, 1.94116816e-04, 5.33983926e-04],
               [1.46890350e-03, 4.04068502e-03, 1.11153126e-02, 3.05763111e-02, 8.41108039e-02, 2.31375217e-01, 6.36473358e-01],],
             
              [[5.43563694e-10, 1.54552393e-09, 4.39442394e-09, 1.24948123e-08, 3.55267176e-08, 1.01013732e-07, 2.87215215e-07],
               [8.16643990e-07, 2.32199136e-06, 6.60217120e-06, 1.87719961e-05, 5.33749480e-05, 1.51762491e-04, 4.31510591e-04],
               [1.22692157e-03, 3.48852738e-03, 9.91902873e-03, 2.82029454e-02, 8.01905394e-02, 2.28007153e-01, 6.48299158e-01],],
             
              [[2.85505758e-10, 8.39077641e-10, 2.46598408e-09, 7.24734939e-09, 2.12993161e-08, 6.25969250e-08, 1.83967856e-07],
               [5.40665155e-07, 1.58897910e-06, 4.66986830e-06, 1.37243542e-05, 4.03348895e-05, 1.18540884e-04, 3.48381785e-04],
               [1.02386903e-03, 3.00905434e-03, 8.84342752e-03, 2.59899981e-02, 7.63827339e-02, 2.24483356e-01, 6.59737885e-01],],
             
              [[1.49833826e-10, 4.55153887e-10, 1.38263490e-09, 4.20008828e-09, 1.27587114e-08, 3.87574559e-08, 1.17735141e-07],
               [3.57646968e-07, 1.08643928e-06, 3.30030957e-06, 1.00254128e-05, 3.04546193e-05, 9.25129279e-05, 2.81029323e-04],
               [8.53691367e-04, 2.59327446e-03, 7.87772890e-03, 2.39303261e-02, 7.26941526e-02, 2.20825255e-01, 6.70807600e-01],],
             
              [[7.85695883e-11, 2.46697551e-10, 7.74595887e-10, 2.43213716e-09, 7.63655716e-09, 2.39777229e-08, 7.52869980e-08],
               [2.36390207e-07, 7.42236580e-07, 2.33052378e-06, 7.31750515e-06, 2.29759917e-05, 7.21415418e-05, 2.26514821e-04],
               [7.11226312e-04, 2.23314716e-03, 7.01181078e-03, 2.20159814e-02, 6.91277757e-02, 2.17051983e-01, 6.81514263e-01],],
             
              [[4.11694637e-11, 1.33612371e-10, 4.33630493e-10, 1.40732292e-09, 4.56736027e-09, 1.48230210e-08, 4.81071787e-08],
               [1.56128209e-07, 5.06706158e-07, 1.64447647e-06, 5.33702359e-06, 1.73209719e-05, 5.62141213e-05, 1.82439369e-04],
               [5.92093158e-04, 1.92159344e-03, 6.23643352e-03, 2.02397965e-02, 6.56872243e-02, 2.13183731e-01, 6.91874325e-01],],
            ]]]])
        ));
        //
        // backward
        //
        // 
        $dResultValiable = $K->ones($resultValiable->shape());
        [$dOutputs,$dSalt] = $resultValiable->creator()->backward([$dResultValiable]);
        $this->assertEquals($outputsVariable->shape(),$dOutputs->shape());
        $this->assertEquals($resultValiable->shape(),$dSalt->shape());
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dOutputs),$K->ndarray($salt)
        ));
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dSalt),$K->ndarray($outputs)
        ));

        $copydOutputs = $K->copy($dOutputs);
        $dInputs = $outputsVariable->creator()->backward([$dOutputs]);
        // 2 batch
        $this->assertCount(2,$dInputs);
        //echo "dQuery: ".$mo->shapeToString($dInputs[0]->shape())."\n";
        //echo "dValue: ".$mo->shapeToString($dInputs[1]->shape())."\n";
        $this->assertEquals($full_query_shape,$dInputs[0]->shape());
        $this->assertEquals($full_value_shape,$dInputs[1]->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());

        //echo "dQuery: ".$mo->toString($dInputs[0],format:'%12.7e',indent:true)."\n";
        //echo "dValue: ".$mo->toString($dInputs[1],format:'%12.7e',indent:true)."\n";

        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dInputs[0]),
            $mo->array([
              [[[ 329.7038,   329.7038,   329.7038,   329.7038,   329.7038 ],
                [1095.3868,  1095.3868,  1095.3868,  1095.3868,  1095.3868 ],
                [1707.2876,  1707.2876,  1707.2876,  1707.2876,  1707.2876 ],
                [2121.2153,  2121.2153,  2121.2153,  2121.2153,  2121.2153 ],
                [2342.534 ,  2342.534 ,  2342.534 ,  2342.534 ,  2342.534  ],
                [2408.4746,  2408.4746,  2408.4746,  2408.4746,  2408.4746 ],],
             
               [[2366.8394,  2366.8394,  2366.8394,  2366.8394,  2366.8394 ],
                [2261.113 ,  2261.113 ,  2261.113 ,  2261.113 ,  2261.113  ],
                [2124.1057,  2124.1057,  2124.1057,  2124.1057,  2124.1057 ],
                [1977.396 ,  1977.396 ,  1977.396 ,  1977.396 ,  1977.396  ],
                [1833.6078,  1833.6078,  1833.6078,  1833.6078,  1833.6078 ],
                [1699.174 ,  1699.174 ,  1699.174 ,  1699.174 ,  1699.174  ],],
             
               [[1576.709 ,  1576.709 ,  1576.709 ,  1576.709 ,  1576.709  ],
                [1466.6049,  1466.6049,  1466.6049,  1466.6049,  1466.6049 ],
                [1368.2327,  1368.2327,  1368.2327,  1368.2327,  1368.2327 ],
                [1280.4333,  1280.4333,  1280.4333,  1280.4333,  1280.4333 ],
                [1201.9622,  1201.9622,  1201.9622,  1201.9622,  1201.9622 ],
                [1131.6251,  1131.6251,  1131.6251,  1131.6251,  1131.6251 ],]],
             
             
              [[[1068.2324 , 1068.2324 , 1068.2324 , 1068.2324 , 1068.2324 ],
                [1011.01855, 1011.01855, 1011.01855, 1011.01855, 1011.01855],
                [ 959.124  ,  959.124  ,  959.124  ,  959.124  ,  959.124  ],
                [ 911.5576 ,  911.5576 ,  911.5576 ,  911.5576 ,  911.5576 ],
                [ 868.2285 ,  868.2285 ,  868.2285 ,  868.2285 ,  868.2285 ],
                [ 828.21484,  828.21484,  828.21484,  828.21484,  828.21484],],
             
               [[ 791.4219 ,  791.4219 ,  791.4219 ,  791.4219 ,  791.4219 ],
                [ 757.3457 ,  757.3457 ,  757.3457 ,  757.3457 ,  757.3457 ],
                [ 725.64453,  725.64453,  725.64453,  725.64453,  725.64453],
                [ 696.1748 ,  696.1748 ,  696.1748 ,  696.1748 ,  696.1748 ],
                [ 668.6953 ,  668.6953 ,  668.6953 ,  668.6953 ,  668.6953 ],
                [ 642.9072 ,  642.9072 ,  642.9072 ,  642.9072 ,  642.9072 ],],
             
               [[ 618.6006 ,  618.6006 ,  618.6006 ,  618.6006 ,  618.6006 ],
                [ 595.87305,  595.87305,  595.87305,  595.87305,  595.87305],
                [ 574.4697 ,  574.4697 ,  574.4697 ,  574.4697 ,  574.4697 ],
                [ 554.08594,  554.08594,  554.08594,  554.08594,  554.08594],
                [ 534.76855,  534.76855,  534.76855,  534.76855,  534.76855],
                [ 516.7656 ,  516.7656 ,  516.7656 ,  516.7656 ,  516.7656 ],]],
            ])
        ));
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dInputs[1]),
            $mo->array([
             [[[-2.00001236e+02, -2.00001236e+02, -2.00001236e+02, -2.00001236e+02, -2.00001236e+02],
               [-2.28900177e+02, -2.28900177e+02, -2.28900177e+02, -2.28900177e+02, -2.28900177e+02],
               [-2.63721863e+02, -2.63721863e+02, -2.63721863e+02, -2.63721863e+02, -2.63721863e+02],
               [-3.05927551e+02, -3.05927551e+02, -3.05927551e+02, -3.05927551e+02, -3.05927551e+02],
               [-3.57324677e+02, -3.57324677e+02, -3.57324677e+02, -3.57324677e+02, -3.57324677e+02],
               [-4.20090057e+02, -4.20090057e+02, -4.20090057e+02, -4.20090057e+02, -4.20090057e+02],
               [-4.96712189e+02, -4.96712189e+02, -4.96712189e+02, -4.96712189e+02, -4.96712189e+02],],
            
              [[-5.89752563e+02, -5.89752563e+02, -5.89752563e+02, -5.89752563e+02, -5.89752563e+02],
               [-7.01224365e+02, -7.01224365e+02, -7.01224365e+02, -7.01224365e+02, -7.01224365e+02],
               [-8.31167175e+02, -8.31167175e+02, -8.31167175e+02, -8.31167175e+02, -8.31167175e+02],
               [-9.74671509e+02, -9.74671509e+02, -9.74671509e+02, -9.74671509e+02, -9.74671509e+02],
               [-1.11584619e+03, -1.11584619e+03, -1.11584619e+03, -1.11584619e+03, -1.11584619e+03],
               [-1.21598657e+03, -1.21598657e+03, -1.21598657e+03, -1.21598657e+03, -1.21598657e+03],
               [-1.19071362e+03, -1.19071362e+03, -1.19071362e+03, -1.19071362e+03, -1.19071362e+03],],
            
              [[-8.66423096e+02, -8.66423096e+02, -8.66423096e+02, -8.66423096e+02, -8.66423096e+02],
               [ 1.01944336e+02,  1.01944336e+02,  1.01944336e+02,  1.01944336e+02,  1.01944336e+02],
               [ 2.38506299e+03,  2.38506299e+03,  2.38506299e+03,  2.38506299e+03,  2.38506299e+03],
               [ 7.26668066e+03,  7.26668066e+03,  7.26668066e+03,  7.26668066e+03,  7.26668066e+03],
               [ 1.71774297e+04,  1.71774297e+04,  1.71774297e+04,  1.71774297e+04,  1.71774297e+04],
               [ 3.66825078e+04,  3.66825078e+04,  3.66825078e+04,  3.66825078e+04,  3.66825078e+04],
               [ 7.43048516e+04,  7.43048516e+04,  7.43048516e+04,  7.43048516e+04,  7.43048516e+04],]],
            
            
             [[[-8.01573694e-01, -8.01573694e-01, -8.01573694e-01, -8.01573694e-01, -8.01573694e-01],
               [-1.46815979e+00, -1.46815979e+00, -1.46815979e+00, -1.46815979e+00, -1.46815979e+00],
               [-2.69048810e+00, -2.69048810e+00, -2.69048810e+00, -2.69048810e+00, -2.69048810e+00],
               [-4.93355703e+00, -4.93355703e+00, -4.93355703e+00, -4.93355703e+00, -4.93355703e+00],
               [-9.05333328e+00, -9.05333328e+00, -9.05333328e+00, -9.05333328e+00, -9.05333328e+00],
               [-1.66277523e+01, -1.66277523e+01, -1.66277523e+01, -1.66277523e+01, -1.66277523e+01],
               [-3.05699577e+01, -3.05699577e+01, -3.05699577e+01, -3.05699577e+01, -3.05699577e+01],],
            
              [[-5.62655830e+01, -5.62655830e+01, -5.62655830e+01, -5.62655830e+01, -5.62655830e+01],
               [-1.03683624e+02, -1.03683624e+02, -1.03683624e+02, -1.03683624e+02, -1.03683624e+02],
               [-1.91278214e+02, -1.91278214e+02, -1.91278214e+02, -1.91278214e+02, -1.91278214e+02],
               [-3.53148315e+02, -3.53148315e+02, -3.53148315e+02, -3.53148315e+02, -3.53148315e+02],
               [-6.51902100e+02, -6.51902100e+02, -6.51902100e+02, -6.51902100e+02, -6.51902100e+02],
               [-1.20072424e+03, -1.20072424e+03, -1.20072424e+03, -1.20072424e+03, -1.20072424e+03],
               [-2.19728833e+03, -2.19728833e+03, -2.19728833e+03, -2.19728833e+03, -2.19728833e+03],],
            
              [[-3.96043286e+03, -3.96043286e+03, -3.96043286e+03, -3.96043286e+03, -3.96043286e+03],
               [-6.90376465e+03, -6.90376465e+03, -6.90376465e+03, -6.90376465e+03, -6.90376465e+03],
               [-1.11547676e+04, -1.11547676e+04, -1.11547676e+04, -1.11547676e+04, -1.11547676e+04],
               [-1.46988848e+04, -1.46988848e+04, -1.46988848e+04, -1.46988848e+04, -1.46988848e+04],
               [-6.00649219e+03, -6.00649219e+03, -6.00649219e+03, -6.00649219e+03, -6.00649219e+03],
               [ 6.29803516e+04,  6.29803516e+04,  6.29803516e+04,  6.29803516e+04,  6.29803516e+04],
               [ 3.71924500e+05,  3.71924500e+05,  3.71924500e+05,  3.71924500e+05,  3.71924500e+05],]],

            ])
        ));

    }

    public function testCausalMask()
    {
        $num_heads = 8;
        $key_dim = 4;
        #$full_query_shape = [2, 6, 16];
        #$full_value_shape = [2, 7, 16];
        #$full_query_shape = [2, 3, 6];
        #$full_value_shape = [2, 5, 6];
        $full_query_shape = [2, 6, 5];
        $full_value_shape = [2, 7, 5];
        $tmp = $full_query_shape;
        $tSeq = array_splice($tmp,1,-1);
        [$batches,$dim] = $tmp;
        $tmp = $full_value_shape;
        $sSeq = array_splice($tmp,1,-1);
        [$batches,$dim] = $tmp;

        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new MultiHeadAttention(
            $K,
            $num_heads, // num_heads
            $key_dim,   // key_dim
            kernel_initializer:'ones',
            bias_initializer:'zeros',
        );

        $salt_q = $mo->la()->range(array_product($full_query_shape),dtype:NDArray::float32)
                ->reshape($full_query_shape);
        $salt_v = $mo->la()->range(array_product($full_value_shape),dtype:NDArray::float32)
                ->reshape($full_value_shape);
        $salt_q = $K->array($salt_q);
        $salt_v = $K->array($salt_v);
        //$query = $g->Variable($la->randomNormal($full_query_shape,mean:0,scale:1));
        //$value = $g->Variable($la->randomNormal($full_value_shape,mean:0,scale:1));
        //echo "query: ".$mo->toString($dInputs[0],format:'%12.7e',indent:true)."\n";
        $query = $g->Variable($K->scale(1/array_product($full_query_shape),$K->increment($salt_q,1)));
        $value = $g->Variable($K->scale(1/array_product($full_value_shape),$K->increment($salt_v,1)));
        //$query = $g->Variable($K->increment(
        //        $K->scale(0.5, $K->array($mo->la()->range(array_product($full_query_shape),dtype:NDArray::float32)
        //        ->reshape($full_query_shape))),
        //    1,
        //));
        //$value = $g->Variable($K->increment(
        //        $K->scale(0.2, $K->array($mo->la()->range(array_product($full_value_shape),dtype:NDArray::float32)
        //        ->reshape($full_value_shape))),
        //    1,
        //));
        //echo 'query:'.$mo->toString($query,format:'%12.7f',indent:true)."\n";
        //echo 'value:'.$mo->toString($value,format:'%12.7f',indent:true)."\n";
        $inputs = [
            $query,
            $value,
        ];

        $layer->build($inputs,
            //sampleWeights:[
            //    $K->ones([$num_heads,  $key_dim, $dim]),    // query_dense/kernel
            //    $K->zeros([$num_heads, $key_dim]),          // query_dense/bias
            //    $K->ones([$num_heads,  $key_dim, $dim]),    // key_dense/kernel
            //    $K->zeros([$num_heads, $key_dim]),          // key_dense/bias
            //    $K->ones([$num_heads,  $key_dim, $dim]),    // value_dense/kernel
            //    $K->zeros([$num_heads, $key_dim]),          // value_dense/bias
            //    $K->ones([$num_heads,  $key_dim, $dim]),    // output_dense/kernel
            //    $K->zeros([$dim]),                          // output_dense/bias
            //]
        );

        //
        // forward
        //
        //echo "query: ".$mo->toString($query,format:'%12.7f',indent:true)."\n";
        //////////////echo "key: ".$mo->toString($key,indent:true)."\n";
        //echo "value: ".$mo->toString($value,format:'%12.7f',indent:true)."\n";

        $salt = $g->Variable($salt_q);
        $copyInputs = [$K->copy($query),$K->copy($value)];
        [$outputsVariable,$scores,$resultValiable] = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$layer,$inputs,$salt) {
                [$outputsVariable,$scores] = $layer->forward(
                    $inputs,
                    training:true,
                    returnAttentionScores:true,
                    useCausalMask:true,
                );
                $resultValiable = $g->mul($outputsVariable,$salt);
                return [$outputsVariable,$scores,$resultValiable];
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        //echo 'query:'.$mo->toString($query,format:'%12.7f',indent:true)."\n";
        //echo 'value:'.$mo->toString($value,format:'%12.7f',indent:true)."\n";
        //echo 'outputs:'.$mo->toString($outputs,format:'%12.7f',indent:true)."\n";
        //echo 'scores:'.$mo->toString($scores,format:'%12.7e',indent:true)."\n";
        //echo 'kernel:'.$mo->toString($layer->getParams()[0],format:'%14.7f',indent:true)."\n";
        $this->assertEquals(array_merge([$batches, $num_heads], $tSeq, $sSeq),$scores->shape());
        $this->assertEquals(array_merge([$batches], $tSeq, [$dim]),$outputs->shape());
        $this->assertEquals($copyInputs[0]->toArray(),$inputs[0]->toArray());
        $this->assertEquals($copyInputs[1]->toArray(),$inputs[1]->toArray());

        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($outputs),
            $mo->array([
               [[  6.8571444,   6.8571444,   6.8571444,   6.8571444,   6.8571444],
                [ 13.906834 ,  13.906834 ,  13.906834 ,  13.906834 ,  13.906834 ],
                [ 23.660307 ,  23.660307 ,  23.660307 ,  23.660307 ,  23.660307 ],
                [ 35.827053 ,  35.827053 ,  35.827053 ,  35.827053 ,  35.827053 ],
                [ 48.73391  ,  48.73391  ,  48.73391  ,  48.73391  ,  48.73391  ],
                [ 61.341904 ,  61.341904 ,  61.341904 ,  61.341904 ,  61.341904 ],],
              
               [[ 86.85714,    86.85714,    86.85714,    86.85714,    86.85714  ],
                [ 97.20768,    97.20768,    97.20768,    97.20768,    97.20768  ],
                [108.77219,   108.77219,   108.77219,   108.77219,   108.77219  ],
                [120.447  ,   120.447  ,   120.447  ,   120.447  ,   120.447    ],
                [132.06229,   132.06229,   132.06229,   132.06229,   132.06229  ],
                [143.62624,   143.62624,   143.62624,   143.62624,   143.62624  ],],
            ])
        ));

        //echo $mo->toString($K->slice($scores,[0,0],[2,1]),format:'%12.7e',indent:true)."\n";
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($K->slice($scores,[0,0],[2,1])),
            $mo->array([
             [[[1.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
               [3.8315210e-01, 6.1684793e-01, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
               [1.2709263e-01, 2.7553806e-01, 5.9736925e-01, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
               [2.6788950e-02, 7.8211628e-02, 2.2834253e-01, 6.6665679e-01, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
               [3.1240571e-03, 1.2282558e-02, 4.8290145e-02, 1.8985772e-01, 7.4644548e-01, 0.0000000e+00, 0.0000000e+00],
               [1.9497830e-04, 1.0323110e-03, 5.4655615e-03, 2.8937357e-02, 1.5320852e-01, 8.1116128e-01, 0.0000000e+00],]],
             
             [[[1.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
               [9.4327442e-02, 9.0567255e-01, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
               [5.5216197e-03, 7.1392700e-02, 9.2308575e-01, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
               [1.7856366e-04, 3.1090998e-03, 5.4134872e-02, 9.4257748e-01, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
               [3.1672921e-06, 7.4264957e-05, 1.7413282e-03, 4.0829647e-02, 9.5735162e-01, 0.0000000e+00, 0.0000000e+00],
               [3.0851325e-08, 9.7414568e-07, 3.0759184e-05, 9.7123248e-04, 3.0667139e-02, 9.6832991e-01, 0.0000000e+00],]],
            ])
        ));
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($K->slice($scores,[0,-1],[2,1])),
            $mo->array([
             [[[1.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
               [3.8315210e-01, 6.1684793e-01, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
               [1.2709263e-01, 2.7553806e-01, 5.9736925e-01, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
               [2.6788950e-02, 7.8211628e-02, 2.2834253e-01, 6.6665679e-01, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
               [3.1240571e-03, 1.2282558e-02, 4.8290145e-02, 1.8985772e-01, 7.4644548e-01, 0.0000000e+00, 0.0000000e+00],
               [1.9497830e-04, 1.0323110e-03, 5.4655615e-03, 2.8937357e-02, 1.5320852e-01, 8.1116128e-01, 0.0000000e+00],]],
             
             [[[1.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
               [9.4327442e-02, 9.0567255e-01, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
               [5.5216197e-03, 7.1392700e-02, 9.2308575e-01, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
               [1.7856366e-04, 3.1090998e-03, 5.4134872e-02, 9.4257748e-01, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
               [3.1672921e-06, 7.4264957e-05, 1.7413282e-03, 4.0829647e-02, 9.5735162e-01, 0.0000000e+00, 0.0000000e+00],
               [3.0851325e-08, 9.7414568e-07, 3.0759184e-05, 9.7123248e-04, 3.0667139e-02, 9.6832991e-01, 0.0000000e+00],]],
            ])
        ));

        //
        // backward
        // 
        $dResultValiable = $K->ones($resultValiable->shape());
        [$dOutputs,$dSalt] = $resultValiable->creator()->backward([$dResultValiable]);
        $this->assertEquals($outputsVariable->shape(),$dOutputs->shape());
        $this->assertEquals($resultValiable->shape(),$dSalt->shape());
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dOutputs),$K->ndarray($salt)
        ));
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dSalt),$K->ndarray($outputs)
        ));

        $copydOutputs = $K->copy($dOutputs);
        $dInputs = $outputsVariable->creator()->backward([$dOutputs]);
        // 2 batch
        $this->assertCount(2,$dInputs);
        //echo "dQuery: ".$mo->shapeToString($dInputs[0]->shape())."\n";
        //echo "dValue: ".$mo->shapeToString($dInputs[1]->shape())."\n";
        $this->assertEquals($full_query_shape,$dInputs[0]->shape());
        $this->assertEquals($full_value_shape,$dInputs[1]->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());

        //echo "dQuery: ".$mo->toString($dInputs[0],format:'%12.7e',indent:true)."\n";
        //echo "dValue: ".$mo->toString($dInputs[1],format:'%12.7e',indent:true)."\n";

        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dInputs[0]),
            $mo->array([
               [[  0.0    ,    0.0    ,    0.0    ,    0.0    ,    0.0     ],
                [ 67.52758,   67.52758,   67.52758,   67.52758,   67.52758 ],
                [246.5154 ,  246.5154 ,  246.5154 ,  246.5154 ,  246.5154  ],
                [392.6947 ,  392.6947 ,  392.6947 ,  392.6947 ,  392.6947  ],
                [386.83835,  386.83835,  386.83835,  386.83835,  386.83835 ],
                [314.57013,  314.57013,  314.57013,  314.57013,  314.57013 ],],
              
               [[  0.0     ,   0.0     ,   0.0     ,   0.0     ,   0.0     ],
                [129.01859 , 129.01859 , 129.01859 , 129.01859 , 129.01859 ],
                [148.58252 , 148.58252 , 148.58252 , 148.58252 , 148.58252 ],
                [123.67627 , 123.67627 , 123.67627 , 123.67627 , 123.67627 ],
                [ 98.72949 ,  98.72949 ,  98.72949 ,  98.72949 ,  98.72949 ],
                [ 78.574585,  78.574585,  78.574585,  78.574585,  78.574585],],               
            ])
        ));
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dInputs[1]),
            $mo->array([
               [[ 414.42322,  414.42322,  414.42322,  414.42322,  414.42322],
                [ 878.32544,  878.32544,  878.32544,  878.32544,  878.32544],
                [1579.6554 , 1579.6554 , 1579.6554 , 1579.6554 , 1579.6554 ],
                [2534.3342 , 2534.3342 , 2534.3342 , 2534.3342 , 2534.3342 ],
                [3650.671  , 3650.671  , 3650.671  , 3650.671  , 3650.671  ],
                [4862.592  , 4862.592  , 4862.592  , 4862.592  , 4862.592  ],
                [   0.     ,    0.     ,    0.     ,    0.     ,    0.     ],],
              
               [[4379.223 ,  4379.223 ,  4379.223 ,  4379.223 ,  4379.223  ],
                [5747.147 ,  5747.147 ,  5747.147 ,  5747.147 ,  5747.147  ],
                [6749.2715,  6749.2715,  6749.2715,  6749.2715,  6749.2715 ],
                [7585.7544,  7585.7544,  7585.7544,  7585.7544,  7585.7544 ],
                [8430.23  ,  8430.23  ,  8430.23  ,  8430.23  ,  8430.23   ],
                [9828.321 ,  9828.321 ,  9828.321 ,  9828.321 ,  9828.321  ],
                [   0.    ,     0.    ,     0.    ,     0.    ,     0.     ],],
            ])
        ));
    }

    public function testMaskBoth()
    {
        $num_heads = 8;
        $key_dim = 4;
        #$full_query_shape = [2, 6, 16];
        #$full_value_shape = [2, 7, 16];
        $full_query_shape = [2, 6, 5];
        $full_value_shape = [2, 7, 5];
        #$full_query_shape = [2, 3, 6, 5];
        #$full_value_shape = [2, 3, 7, 5];
        $query_mask = [
            [True,True,True,False,False,False],
            [True,True,True,True,True,False],
        ];
        $value_mask = [
            [True,True,True,True,False,False,False],
            [True,True,True,True,True,True,False],
        ];
        
        $tmp = $full_query_shape;
        $tSeq = array_splice($tmp,1,-1);
        [$batches,$dim] = $tmp;
        $tmp = $full_value_shape;
        $sSeq = array_splice($tmp,1,-1);
        [$batches,$dim] = $tmp;

        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $la = $K->primaryLA();
        $srvLvl = $K->localMatrixOperator()->service()->serviceLevel();
        $layer = new MultiHeadAttention(
            $K,
            $num_heads, // num_heads
            $key_dim,   // key_dim
            kernel_initializer:'ones',
            bias_initializer:'zeros',
        );
        $salt_q = $mo->la()->range(array_product($full_query_shape),dtype:NDArray::float32)
                ->reshape($full_query_shape);
        $salt_v = $mo->la()->range(array_product($full_value_shape),dtype:NDArray::float32)
                ->reshape($full_value_shape);
        $salt_q = $K->array($salt_q);
        $salt_v = $K->array($salt_v);
        //$query = $g->Variable($la->randomNormal($full_query_shape,mean:0,scale:1));
        //$value = $g->Variable($la->randomNormal($full_value_shape,mean:0,scale:1));
        //echo "query: ".$mo->toString($dInputs[0],format:'%12.7e',indent:true)."\n";
        $query = $g->Variable($K->scale(1/array_product($full_query_shape),$K->increment($salt_q,1)));
        $value = $g->Variable($K->scale(1/array_product($full_value_shape),$K->increment($salt_v,1)));
        $query_mask = $g->Variable($K->array($query_mask,dtype:NDArray::bool));
        $value_mask = $g->Variable($K->array($value_mask,dtype:NDArray::bool));
        //$query = $g->Variable($K->increment(
        //        $K->scale(0.5, $K->array($mo->la()->range(array_product($full_query_shape),dtype:NDArray::float32)
        //        ->reshape($full_query_shape))),
        //    1,
        //));
        //$value = $g->Variable($K->increment(
        //        $K->scale(0.2, $K->array($mo->la()->range(array_product($full_value_shape),dtype:NDArray::float32)
        //        ->reshape($full_value_shape))),
        //    1,
        //));
        //echo 'query:'.$mo->toString($query,format:'%12.7f',indent:true)."\n";
        //echo 'value:'.$mo->toString($value,format:'%12.7f',indent:true)."\n";
        //echo 'query:'.$mo->shapeToString($query->shape())."\n";
        //echo 'value:'.$mo->shapeToString($value->shape())."\n";
        //echo 'query_mask:'.$mo->shapeToString($query_mask->shape())."\n";
        //echo 'value_mask:'.$mo->shapeToString($value_mask->shape())."\n";
        $inputs = [
            $query,
            $value,
        ];

        $layer->build($inputs,
        );

        //
        // forward
        //
        //  batch size 2
        //echo "query: ".$mo->toString($query,format:'%12.7f',indent:true)."\n";
        //////////////echo "key: ".$mo->toString($key,indent:true)."\n";
        //echo "value: ".$mo->toString($value,format:'%12.7f',indent:true)."\n";

        #$query_mask = null;
        #$value_mask = null;
        $salt = $g->Variable($salt_q);
        $copyInputs = [$K->copy($query),$K->copy($value)];
        [$outputsVariable,$scores,$resultValiable] = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$layer,$inputs,$salt,$query_mask,$value_mask) {
                [$outputsVariable,$scores] = $layer->forward(
                    $inputs,
                    training:true,
                    returnAttentionScores:true,
                    mask:[$query_mask,$value_mask],
                );
                $resultValiable = $g->mul($outputsVariable,$salt);
                return [$outputsVariable,$scores,$resultValiable];
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        //echo 'query:'.$mo->toString($query,format:'%12.7f',indent:true)."\n";
        //echo 'value:'.$mo->toString($value,format:'%12.7f',indent:true)."\n";
        //echo 'outputs:'.$mo->toString($outputs,format:'%12.7f',indent:true)."\n";
        //echo 'scores:'.$mo->toString($scores,format:'%12.7e',indent:true)."\n";
        //echo 'kernel:'.$mo->toString($layer->getParams()[0],format:'%14.7f',indent:true)."\n";
        $this->assertEquals(array_merge([$batches, $num_heads], $tSeq, $sSeq),$scores->shape());
        $this->assertEquals(array_merge([$batches], $tSeq, [$dim]),$outputs->shape());
        $this->assertEquals($copyInputs[0]->toArray(),$inputs[0]->toArray());
        $this->assertEquals($copyInputs[1]->toArray(),$inputs[1]->toArray());
        //$this->assertTrue($mo->la()->isclose(
        //    $K->fill([2,8,6,7], 0.14285715),
        //    $K->ndarray($scores)));
        //$this->assertTrue($mo->la()->isclose(
        //    //$K->mul($salt,$K->fill($full_query_shape,512)),
        //    $K->fill($full_query_shape,512),
        //    $K->ndarray($outputs)
        //));
        
        //echo "outputs: ".$mo->toString($outputs,format:'%12.7f',indent:true)."\n";
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($outputs),
            $mo->array([
               [[ 26.528252,  26.528252,  26.528252,  26.528252,  26.528252],
                [ 30.400398,  30.400398,  30.400398,  30.400398,  30.400398],
                [ 33.52552 ,  33.52552 ,  33.52552 ,  33.52552 ,  33.52552 ],
                [ 41.14286 ,  41.14286 ,  41.14286 ,  41.14286 ,  41.14286 ],
                [ 41.14286 ,  41.14286 ,  41.14286 ,  41.14286 ,  41.14286 ],
                [ 41.14286 ,  41.14286 ,  41.14286 ,  41.14286 ,  41.14286 ],],
              
               [[142.1361 ,  142.1361 ,  142.1361 ,  142.1361 ,  142.1361  ],
                [142.67139,  142.67139,  142.67139,  142.67139,  142.67139 ],
                [143.042  ,  143.042  ,  143.042  ,  143.042  ,  143.042   ],
                [143.30368,  143.30368,  143.30368,  143.30368,  143.30368 ],
                [143.49086,  143.49086,  143.49086,  143.49086,  143.49086 ],
                [132.57143,  132.57143,  132.57143,  132.57143,  132.57143 ],],
            ]),
            rtol:($srvLvl==Service::LV_BASIC)?1e-1:1e-4,
            //debug:true,
        ));

        //echo "scores0: ".$mo->toString($K->slice($scores,[0,0],[2,1]),format:'%12.7e',indent:true)."\n";
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($K->slice($scores,[0,0],[2,1])),
            $mo->array([
             [[[1.87497035e-01, 2.24154279e-01, 2.67978311e-01, 3.20370287e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
               [1.06671929e-01, 1.71734333e-01, 2.76480168e-01, 4.45113599e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
               [5.53755946e-02, 1.20054819e-01, 2.60280043e-01, 5.64289570e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
               [1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01],
               [1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01],
               [1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01],]],
             
             [[[4.66638303e-05, 3.32704338e-04, 2.37212842e-03, 1.69127975e-02, 1.20585077e-01, 8.59750569e-01, 0.00000000e+00],
               [1.09792027e-05, 1.05415376e-04, 1.01213378e-03, 9.71783325e-03, 9.33045000e-02, 8.95849168e-01, 0.00000000e+00],
               [2.55331861e-06, 3.30135626e-05, 4.26855200e-04, 5.51907532e-03, 7.13598132e-02, 9.22658682e-01, 0.00000000e+00],
               [5.88987461e-07, 1.02552840e-05, 1.78562434e-04, 3.10906675e-03, 5.41340895e-02, 9.42567468e-01, 0.00000000e+00],
               [1.35080356e-07, 3.16729142e-06, 7.42650882e-05, 1.74132455e-03, 4.08296399e-02, 9.57351446e-01, 0.00000000e+00],
               [3.20762188e-29, 3.20762188e-29, 2.00000003e-01, 2.00000003e-01, 2.00000003e-01, 2.00000003e-01, 2.00000003e-01],]],
            ]),
            rtol:($srvLvl==Service::LV_BASIC)?1e+0:1e-4,
            //debug:true,
        ));

        //echo "scores: ".$mo->toString($K->slice($scores,[-1,-1,-1],[1,1,1]),format:'%12.7e',indent:true)."\n";
        //echo "scores: ".$mo->shapeToString($K->slice($scores,[-1,-1,-1],[1,1,1])->shape())."\n";
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($K->slice($scores,[0,-1],[2,1])),
            $mo->array([
             [[[1.87497035e-01, 2.24154279e-01, 2.67978311e-01, 3.20370287e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
               [1.06671929e-01, 1.71734333e-01, 2.76480168e-01, 4.45113599e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
               [5.53755946e-02, 1.20054819e-01, 2.60280043e-01, 5.64289570e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
               [1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01],
               [1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01],
               [1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01],]],
             
             [[[4.66638303e-05, 3.32704338e-04, 2.37212842e-03, 1.69127975e-02, 1.20585077e-01, 8.59750569e-01, 0.00000000e+00],
               [1.09792027e-05, 1.05415376e-04, 1.01213378e-03, 9.71783325e-03, 9.33045000e-02, 8.95849168e-01, 0.00000000e+00],
               [2.55331861e-06, 3.30135626e-05, 4.26855200e-04, 5.51907532e-03, 7.13598132e-02, 9.22658682e-01, 0.00000000e+00],
               [5.88987461e-07, 1.02552840e-05, 1.78562434e-04, 3.10906675e-03, 5.41340895e-02, 9.42567468e-01, 0.00000000e+00],
               [1.35080356e-07, 3.16729142e-06, 7.42650882e-05, 1.74132455e-03, 4.08296399e-02, 9.57351446e-01, 0.00000000e+00],
               [3.20762188e-29, 3.20762188e-29, 2.00000003e-01, 2.00000003e-01, 2.00000003e-01, 2.00000003e-01, 2.00000003e-01],]],
            ]),
            rtol:($srvLvl==Service::LV_BASIC)?1e+0:1e-4,
            //debug:true,
        ));
        //
        // backward
        //
        // 
        $dResultValiable = $K->ones($resultValiable->shape());
        [$dOutputs,$dSalt] = $resultValiable->creator()->backward([$dResultValiable]);
        $this->assertEquals($outputsVariable->shape(),$dOutputs->shape());
        $this->assertEquals($resultValiable->shape(),$dSalt->shape());
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dOutputs),$K->ndarray($salt)
        ));
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dSalt),$K->ndarray($outputs)
        ));

        $copydOutputs = $K->copy($dOutputs);
        $dInputs = $outputsVariable->creator()->backward([$dOutputs]);
        // 2 batch
        $this->assertCount(2,$dInputs);
        //echo "dQuery: ".$mo->shapeToString($dInputs[0]->shape())."\n";
        //echo "dValue: ".$mo->shapeToString($dInputs[1]->shape())."\n";
        $this->assertEquals($full_query_shape,$dInputs[0]->shape());
        $this->assertEquals($full_value_shape,$dInputs[1]->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());

        //echo "dQuery: ".$mo->toString($dInputs[0],format:'%12.7e',indent:true)."\n";
        //echo "dValue: ".$mo->toString($dInputs[1],format:'%12.7e',indent:true)."\n";

        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dInputs[0]),
            $mo->array([
               [[  99.330315,   99.330315,   99.330315,   99.330315,   99.330315],
                [ 297.12323 ,  297.12323 ,  297.12323 ,  297.12323 ,  297.12323 ],
                [ 389.20917 ,  389.20917 ,  389.20917 ,  389.20917 ,  389.20917 ],
                [2775.5093  , 2775.5093  , 2775.5093  , 2775.5093  , 2775.5093  ],
                [3591.8362  , 3591.8362  , 3591.8362  , 3591.8362  , 3591.8362  ],
                [4408.163   , 4408.163   , 4408.163   , 4408.163   , 4408.163   ],],
              
               [[ 247.50195,   247.50195,   247.50195,   247.50195,   247.50195 ],
                [ 195.90976,   195.90976,   195.90976,   195.90976,   195.90976 ],
                [ 155.72949,   155.72949,   155.72949,   155.72949,   155.72949 ],
                [ 124.00012,   124.00012,   124.00012,   124.00012,   124.00012 ],
                [  98.75073,    98.75073,    98.75073,    98.75073,    98.75073 ],
                [4653.039  ,  4653.039  ,  4653.039  ,  4653.039  ,  4653.039   ],],
            ]),
            rtol:($srvLvl==Service::LV_BASIC)?1e+1:1e-4,
            //debug:true,
        ));
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dInputs[1]),
            $mo->array([
              [[-4.9332588e+03, -4.9332588e+03, -4.9332588e+03, -4.9332588e+03, -4.9332588e+03],
               [-2.6076963e+03, -2.6076963e+03, -2.6076963e+03, -2.6076963e+03, -2.6076963e+03],
               [ 1.3684351e+02,  1.3684351e+02,  1.3684351e+02,  1.3684351e+02,  1.3684351e+02],
               [ 3.9983945e+03,  3.9983945e+03,  3.9983945e+03,  3.9983945e+03,  3.9983945e+03],
               [ 3.6419038e+03,  3.6419038e+03,  3.6419038e+03,  3.6419038e+03,  3.6419038e+03],
               [ 5.7752368e+03,  5.7752368e+03,  5.7752368e+03,  5.7752368e+03,  5.7752368e+03],
               [ 7.9085723e+03,  7.9085723e+03,  7.9085723e+03,  7.9085723e+03,  7.9085723e+03],],
             
              [[-2.9573274e+00, -2.9573274e+00, -2.9573274e+00, -2.9573274e+00, -2.9573274e+00],
               [-1.9088331e+01, -1.9088331e+01, -1.9088331e+01, -1.9088331e+01, -1.9088331e+01],
               [-1.0892528e+04, -1.0892528e+04, -1.0892528e+04, -1.0892528e+04, -1.0892528e+04],
               [-5.2124438e+03, -5.2124438e+03, -5.2124438e+03, -5.2124438e+03, -5.2124438e+03],
               [-1.1397036e+03, -1.1397036e+03, -1.1397036e+03, -1.1397036e+03, -1.1397036e+03],
               [ 4.5568398e+04,  4.5568398e+04,  4.5568398e+04,  4.5568398e+04,  4.5568398e+04],
               [ 1.4418279e+04,  1.4418279e+04,  1.4418279e+04,  1.4418279e+04,  1.4418279e+04],],
            ]),
            rtol:($srvLvl==Service::LV_BASIC)?1e+0:1e-4,
            //debug:true,
        ));

    }

    public function testMaskCausalAndBoth()
    {
        $num_heads = 8;
        $key_dim = 4;
        #$full_query_shape = [2, 6, 16];
        #$full_value_shape = [2, 7, 16];
        $full_query_shape = [2, 6, 5];
        $full_value_shape = [2, 7, 5];
        #$full_query_shape = [2, 3, 6, 5];
        #$full_value_shape = [2, 3, 7, 5];
        $query_mask = [
            [True,True,True,False,False,False],
            [True,True,True,True,True,False],
        ];
        $value_mask = [
            [True,True,True,True,False,False,False],
            [True,True,True,True,True,True,False],
        ];
        
        $tmp = $full_query_shape;
        $tSeq = array_splice($tmp,1,-1);
        [$batches,$dim] = $tmp;
        $tmp = $full_value_shape;
        $sSeq = array_splice($tmp,1,-1);
        [$batches,$dim] = $tmp;

        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $la = $K->primaryLA();
        $srvLvl = $K->localMatrixOperator()->service()->serviceLevel();
        $layer = new MultiHeadAttention(
            $K,
            $num_heads, // num_heads
            $key_dim,   // key_dim
            kernel_initializer:'ones',
            bias_initializer:'zeros',
        );
        $salt_q = $mo->la()->range(array_product($full_query_shape),dtype:NDArray::float32)
                ->reshape($full_query_shape);
        $salt_v = $mo->la()->range(array_product($full_value_shape),dtype:NDArray::float32)
                ->reshape($full_value_shape);
        $salt_q = $K->array($salt_q);
        $salt_v = $K->array($salt_v);
        //$query = $g->Variable($la->randomNormal($full_query_shape,mean:0,scale:1));
        //$value = $g->Variable($la->randomNormal($full_value_shape,mean:0,scale:1));
        //echo "query: ".$mo->toString($dInputs[0],format:'%12.7e',indent:true)."\n";
        $query = $g->Variable($K->scale(1/array_product($full_query_shape),$K->increment($salt_q,1)));
        $value = $g->Variable($K->scale(1/array_product($full_value_shape),$K->increment($salt_v,1)));
        $query_mask = $g->Variable($K->array($query_mask,dtype:NDArray::bool));
        $value_mask = $g->Variable($K->array($value_mask,dtype:NDArray::bool));
        //$query = $g->Variable($K->increment(
        //        $K->scale(0.5, $K->array($mo->la()->range(array_product($full_query_shape),dtype:NDArray::float32)
        //        ->reshape($full_query_shape))),
        //    1,
        //));
        //$value = $g->Variable($K->increment(
        //        $K->scale(0.2, $K->array($mo->la()->range(array_product($full_value_shape),dtype:NDArray::float32)
        //        ->reshape($full_value_shape))),
        //    1,
        //));
        //echo 'query:'.$mo->toString($query,format:'%12.7f',indent:true)."\n";
        //echo 'value:'.$mo->toString($value,format:'%12.7f',indent:true)."\n";
        //echo 'query:'.$mo->shapeToString($query->shape())."\n";
        //echo 'value:'.$mo->shapeToString($value->shape())."\n";
        //echo 'query_mask:'.$mo->shapeToString($query_mask->shape())."\n";
        //echo 'value_mask:'.$mo->shapeToString($value_mask->shape())."\n";
        $inputs = [
            $query,
            $value,
        ];

        $layer->build($inputs,
        );

        //
        // forward
        //
        //  batch size 2
        //echo "query: ".$mo->toString($query,format:'%12.7f',indent:true)."\n";
        //////////////echo "key: ".$mo->toString($key,indent:true)."\n";
        //echo "value: ".$mo->toString($value,format:'%12.7f',indent:true)."\n";

        #$query_mask = null;
        #$value_mask = null;
        $salt = $g->Variable($salt_q);
        $copyInputs = [$K->copy($query),$K->copy($value)];
        [$outputsVariable,$scores,$resultValiable] = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$layer,$inputs,$salt,$query_mask,$value_mask) {
                [$outputsVariable,$scores] = $layer->forward(
                    $inputs,
                    training:true,
                    returnAttentionScores:true,
                    useCausalMask:true,
                    mask:[$query_mask,$value_mask],
                );
                $resultValiable = $g->mul($outputsVariable,$salt);
                return [$outputsVariable,$scores,$resultValiable];
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        //echo 'query:'.$mo->toString($query,format:'%12.7f',indent:true)."\n";
        //echo 'value:'.$mo->toString($value,format:'%12.7f',indent:true)."\n";
        //echo 'outputs:'.$mo->toString($outputs,format:'%12.7f',indent:true)."\n";
        //echo 'scores:'.$mo->toString($scores,format:'%12.7e',indent:true)."\n";
        //echo 'kernel:'.$mo->toString($layer->getParams()[0],format:'%14.7f',indent:true)."\n";
        $this->assertEquals(array_merge([$batches, $num_heads], $tSeq, $sSeq),$scores->shape());
        $this->assertEquals(array_merge([$batches], $tSeq, [$dim]),$outputs->shape());
        $this->assertEquals($copyInputs[0]->toArray(),$inputs[0]->toArray());
        $this->assertEquals($copyInputs[1]->toArray(),$inputs[1]->toArray());
        //$this->assertTrue($mo->la()->isclose(
        //    $K->fill([2,8,6,7], 0.14285715),
        //    $K->ndarray($scores)));
        //$this->assertTrue($mo->la()->isclose(
        //    //$K->mul($salt,$K->fill($full_query_shape,512)),
        //    $K->fill($full_query_shape,512),
        //    $K->ndarray($outputs)
        //));
        
        //echo "outputs: ".$mo->toString($outputs,format:'%12.7f',indent:true)."\n";
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($outputs),
            $mo->array([
               [[  6.8571444,   6.8571444,   6.8571444,   6.8571444,   6.8571444],
                [ 13.906834 ,  13.906834 ,  13.906834 ,  13.906834 ,  13.906834 ],
                [ 23.660307 ,  23.660307 ,  23.660307 ,  23.660307 ,  23.660307 ],
                [ 41.14286  ,  41.14286  ,  41.14286  ,  41.14286  ,  41.14286  ],
                [ 41.14286  ,  41.14286  ,  41.14286  ,  41.14286  ,  41.14286  ],
                [ 41.14286  ,  41.14286  ,  41.14286  ,  41.14286  ,  41.14286  ],],
              
               [[ 86.85714  ,  86.85714  ,  86.85714  ,  86.85714  ,  86.85714  ],
                [ 97.20768  ,  97.20768  ,  97.20768  ,  97.20768  ,  97.20768  ],
                [108.77219  , 108.77219  , 108.77219  , 108.77219  , 108.77219  ],
                [120.447    , 120.447    , 120.447    , 120.447    , 120.447    ],
                [132.06229  , 132.06229  , 132.06229  , 132.06229  , 132.06229  ],
                [132.57143  , 132.57143  , 132.57143  , 132.57143  , 132.57143  ],],
            ]),
            rtol:($srvLvl==Service::LV_BASIC)?1e+0:1e-4,
            //debug:true,
        ));

        //echo "scores0: ".$mo->toString($K->slice($scores,[0,0],[2,1]),format:'%12.7e',indent:true)."\n";
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($K->sum($scores,axis:1)),
            //$K->ndarray($K->slice($scores,[0,0],[2,1])),
            $mo->array([
              [[8.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
               [3.0652165e+00, 4.9347835e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
               [1.0167409e+00, 2.2043045e+00, 4.7789540e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
               [1.1428572e+00, 1.1428572e+00, 1.1428572e+00, 1.1428572e+00, 1.1428572e+00, 1.1428572e+00, 1.1428572e+00],
               [1.1428572e+00, 1.1428572e+00, 1.1428572e+00, 1.1428572e+00, 1.1428572e+00, 1.1428572e+00, 1.1428572e+00],
               [1.1428572e+00, 1.1428572e+00, 1.1428572e+00, 1.1428572e+00, 1.1428572e+00, 1.1428572e+00, 1.1428572e+00],],
             
              [[8.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
               [7.5461954e-01, 7.2453804e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
               [4.4172958e-02, 5.7114166e-01, 7.3846860e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
               [1.4285092e-03, 2.4872797e-02, 4.3307897e-01, 7.5406194e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
               [2.5338335e-05, 5.9411960e-04, 1.3930625e-02, 3.2663718e-01, 7.6588130e+00, 0.0000000e+00, 0.0000000e+00],
               [2.5660975e-28, 2.5660975e-28, 1.6000001e+00, 1.6000001e+00, 1.6000001e+00, 1.6000001e+00, 1.6000001e+00],],
            ]),
            rtol:($srvLvl==Service::LV_BASIC)?1e+0:1e-4,
            //debug:true,
        ));

        //
        // backward
        //
        // 
        $dResultValiable = $K->ones($resultValiable->shape());
        [$dOutputs,$dSalt] = $resultValiable->creator()->backward([$dResultValiable]);
        $this->assertEquals($outputsVariable->shape(),$dOutputs->shape());
        $this->assertEquals($resultValiable->shape(),$dSalt->shape());
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dOutputs),$K->ndarray($salt)
        ));
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dSalt),$K->ndarray($outputs)
        ));

        $copydOutputs = $K->copy($dOutputs);
        $dInputs = $outputsVariable->creator()->backward([$dOutputs]);
        // 2 batch
        $this->assertCount(2,$dInputs);
        //echo "dQuery: ".$mo->shapeToString($dInputs[0]->shape())."\n";
        //echo "dValue: ".$mo->shapeToString($dInputs[1]->shape())."\n";
        $this->assertEquals($full_query_shape,$dInputs[0]->shape());
        $this->assertEquals($full_value_shape,$dInputs[1]->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());

        //echo "dQuery: ".$mo->toString($dInputs[0],format:'%12.7e',indent:true)."\n";
        //echo "dValue: ".$mo->toString($dInputs[1],format:'%12.7e',indent:true)."\n";

        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dInputs[0]),
            $mo->array([
               [[   0.0    ,    0.0    ,    0.0    ,    0.0    ,    0.0    ],
                [  67.52758,   67.52758,   67.52758,   67.52758,   67.52758],
                [ 246.5154 ,  246.5154 ,  246.5154 ,  246.5154 ,  246.5154 ],
                [2775.5093 , 2775.5093 , 2775.5093 , 2775.5093 , 2775.5093 ],
                [3591.8362 , 3591.8362 , 3591.8362 , 3591.8362 , 3591.8362 ],
                [4408.163  , 4408.163  , 4408.163  , 4408.163  , 4408.163  ],],
              
               [[   0.0    ,    0.0    ,    0.0    ,    0.0    ,    0.0    ],
                [ 129.01859,  129.01859,  129.01859,  129.01859,  129.01859],
                [ 148.58252,  148.58252,  148.58252,  148.58252,  148.58252],
                [ 123.67627,  123.67627,  123.67627,  123.67627,  123.67627],
                [  98.72949,   98.72949,   98.72949,   98.72949,   98.72949],
                [4653.039  , 4653.039  , 4653.039  , 4653.039  , 4653.039  ],],
            ]),
            rtol:($srvLvl==Service::LV_BASIC)?1e+1:1e-4,
            //debug:true,
        ));
        //echo $mo->toString($dInputs[1],format:'%12.4f',indent:true)."\n";
        //echo $mo->shapeToString($dInputs[1]->shape())."\n";
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dInputs[1]),
            $mo->array([
               [[-4301.9546 , -4301.9546 , -4301.9546 , -4301.9546 , -4301.9546 ],
                [-1604.6592 , -1604.6592 , -1604.6592 , -1604.6592 , -1604.6592 ],
                [  992.32654,   992.32654,   992.32654,   992.32654,   992.32654],
                [ 1508.5709 ,  1508.5709 ,  1508.5709 ,  1508.5709 ,  1508.5709 ],
                [ 3641.9038 ,  3641.9038 ,  3641.9038 ,  3641.9038 ,  3641.9038 ],
                [ 5775.237  ,  5775.237  ,  5775.237  ,  5775.237  ,  5775.237  ],
                [ 7908.5723 ,  7908.5723 ,  7908.5723 ,  7908.5723 ,  7908.5723 ],],
              
               [[ 4379.2275 ,  4379.2275,   4379.2275,   4379.2275,   4379.2275 ],
                [ 5747.2607 ,  5747.2607,   5747.2607,   5747.2607,   5747.2607 ],
                [-4018.421  , -4018.421 ,  -4018.421 ,  -4018.421 ,  -4018.421  ],
                [ 3163.9092 ,  3163.9092,   3163.9092,   3163.9092,   3163.9092 ],
                [10908.535  , 10908.535 ,  10908.535 ,  10908.535 ,  10908.535  ],
                [ 8121.1396 ,  8121.1396,   8121.1396,   8121.1396,   8121.1396 ],
                [14418.279  , 14418.279 ,  14418.279 ,  14418.279 ,  14418.279  ],],
            ]),
            rtol:($srvLvl==Service::LV_BASIC)?1e+0:1e-4,
            //debug:true,
        ));

    }

    public function testMaskQueryOnly()
    {
        $num_heads = 8;
        $key_dim = 4;
        #$full_query_shape = [2, 6, 16];
        #$full_value_shape = [2, 7, 16];
        $full_query_shape = [2, 6, 5];
        $full_value_shape = [2, 7, 5];
        #$full_query_shape = [2, 3, 6, 5];
        #$full_value_shape = [2, 3, 7, 5];
        $query_mask = [
            [True,True,True,False,False,False],
            [True,True,True,True,True,False],
        ];
        //$value_mask = [
        //    [True,True,True,True,False,False,False],
        //    [True,True,True,True,True,True,False],
        //];
        
        $tmp = $full_query_shape;
        $tSeq = array_splice($tmp,1,-1);
        [$batches,$dim] = $tmp;
        $tmp = $full_value_shape;
        $sSeq = array_splice($tmp,1,-1);
        [$batches,$dim] = $tmp;

        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $la = $K->primaryLA();
        $srvLvl = $K->localMatrixOperator()->service()->serviceLevel();
        $layer = new MultiHeadAttention(
            $K,
            $num_heads, // num_heads
            $key_dim,   // key_dim
            kernel_initializer:'ones',
            bias_initializer:'zeros',
        );
        $salt_q = $mo->la()->range(array_product($full_query_shape),dtype:NDArray::float32)
                ->reshape($full_query_shape);
        $salt_v = $mo->la()->range(array_product($full_value_shape),dtype:NDArray::float32)
                ->reshape($full_value_shape);
        $salt_q = $K->array($salt_q);
        $salt_v = $K->array($salt_v);
        //$query = $g->Variable($la->randomNormal($full_query_shape,mean:0,scale:1));
        //$value = $g->Variable($la->randomNormal($full_value_shape,mean:0,scale:1));
        //echo "query: ".$mo->toString($dInputs[0],format:'%12.7e',indent:true)."\n";
        $query = $g->Variable($K->scale(1/array_product($full_query_shape),$K->increment($salt_q,1)));
        $value = $g->Variable($K->scale(1/array_product($full_value_shape),$K->increment($salt_v,1)));
        $query_mask = $g->Variable($K->array($query_mask,dtype:NDArray::bool));
        $value_mask = null;
        //$value_mask = $g->Variable($K->array($value_mask,dtype:NDArray::bool));
        //$query = $g->Variable($K->increment(
        //        $K->scale(0.5, $K->array($mo->la()->range(array_product($full_query_shape),dtype:NDArray::float32)
        //        ->reshape($full_query_shape))),
        //    1,
        //));
        //$value = $g->Variable($K->increment(
        //        $K->scale(0.2, $K->array($mo->la()->range(array_product($full_value_shape),dtype:NDArray::float32)
        //        ->reshape($full_value_shape))),
        //    1,
        //));
        //echo 'query:'.$mo->toString($query,format:'%12.7f',indent:true)."\n";
        //echo 'value:'.$mo->toString($value,format:'%12.7f',indent:true)."\n";
        //echo 'query:'.$mo->shapeToString($query->shape())."\n";
        //echo 'value:'.$mo->shapeToString($value->shape())."\n";
        //echo 'query_mask:'.$mo->shapeToString($query_mask->shape())."\n";
        //echo 'value_mask:'.$mo->shapeToString($value_mask->shape())."\n";
        $inputs = [
            $query,
            $value,
        ];

        $layer->build($inputs,
        );

        //
        // forward
        //
        //  batch size 2
        //echo "query: ".$mo->toString($query,format:'%12.7f',indent:true)."\n";
        //////////////echo "key: ".$mo->toString($key,indent:true)."\n";
        //echo "value: ".$mo->toString($value,format:'%12.7f',indent:true)."\n";

        #$query_mask = null;
        #$value_mask = null;
        $salt = $g->Variable($salt_q);
        $copyInputs = [$K->copy($query),$K->copy($value)];
        [$outputsVariable,$scores,$resultValiable] = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$layer,$inputs,$salt,$query_mask,$value_mask) {
                [$outputsVariable,$scores] = $layer->forward(
                    $inputs,
                    training:true,
                    returnAttentionScores:true,
                    mask:[$query_mask,$value_mask],
                );
                $resultValiable = $g->mul($outputsVariable,$salt);
                return [$outputsVariable,$scores,$resultValiable];
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        //echo 'query:'.$mo->toString($query,format:'%12.7f',indent:true)."\n";
        //echo 'value:'.$mo->toString($value,format:'%12.7f',indent:true)."\n";
        //echo 'outputs:'.$mo->toString($outputs,format:'%12.7f',indent:true)."\n";
        //echo 'scores:'.$mo->toString($scores,format:'%12.7e',indent:true)."\n";
        //echo 'kernel:'.$mo->toString($layer->getParams()[0],format:'%14.7f',indent:true)."\n";
        $this->assertEquals(array_merge([$batches, $num_heads], $tSeq, $sSeq),$scores->shape());
        $this->assertEquals(array_merge([$batches], $tSeq, [$dim]),$outputs->shape());
        $this->assertEquals($copyInputs[0]->toArray(),$inputs[0]->toArray());
        $this->assertEquals($copyInputs[1]->toArray(),$inputs[1]->toArray());
        //$this->assertTrue($mo->la()->isclose(
        //    $K->fill([2,8,6,7], 0.14285715),
        //    $K->ndarray($scores)));
        //$this->assertTrue($mo->la()->isclose(
        //    //$K->mul($salt,$K->fill($full_query_shape,512)),
        //    $K->fill($full_query_shape,512),
        //    $K->ndarray($outputs)
        //));
        
        //echo "outputs: ".$mo->toString($outputs,format:'%12.7f',indent:true)."\n";
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($outputs),
            $mo->array([
               [[ 49.09698 ,  49.09698 ,  49.09698 ,  49.09698 ,  49.09698 ],
                [ 59.650528,  59.650528,  59.650528,  59.650528,  59.650528],
                [ 66.000854,  66.000854,  66.000854,  66.000854,  66.000854],
                [ 41.14286 ,  41.14286 ,  41.14286 ,  41.14286 ,  41.14286 ],
                [ 41.14286 ,  41.14286 ,  41.14286 ,  41.14286 ,  41.14286 ],
                [ 41.14286 ,  41.14286 ,  41.14286 ,  41.14286 ,  41.14286 ],],
              
               [[153.56422,  153.56422,  153.56422,  153.56422,  153.56422 ],
                [154.09987,  154.09987,  154.09987,  154.09987,  154.09987 ],
                [154.47055,  154.47055,  154.47055,  154.47055,  154.47055 ],
                [154.7322 ,  154.7322 ,  154.7322 ,  154.7322 ,  154.7322  ],
                [154.91945,  154.91945,  154.91945,  154.91945,  154.91945 ],
                [132.57143,  132.57143,  132.57143,  132.57143,  132.57143 ],],               
            ]),
            rtol:($srvLvl==Service::LV_BASIC)?1e-1:1e-4,
            //debug:true,
        ));

        //echo "scores0: ".$mo->toString($K->slice($scores,[0,0],[2,1]),format:'%12.7e',indent:true)."\n";
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($K->slice($scores,[0,0],[2,1])),
            $mo->array([
             [[[7.85065666e-02, 9.38552469e-02, 1.12204731e-01, 1.34141684e-01, 1.60367504e-01, 1.91720679e-01, 2.29203656e-01],
               [2.25635562e-02, 3.63257453e-02, 5.84818795e-02, 9.41517130e-02, 1.51577622e-01, 2.44029343e-01, 3.92870098e-01],
               [5.21136262e-03, 1.12982830e-02, 2.44947895e-02, 5.31049334e-02, 1.15132004e-01, 2.49607325e-01, 5.41151285e-01],
               [1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01],
               [1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01],
               [1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01],]],
             
             [[[6.54485575e-06, 4.66635938e-05, 3.32703930e-04, 2.37211213e-03, 1.69127118e-02, 1.20584704e-01, 8.59744608e-01],
               [1.14350473e-06, 1.09792109e-05, 1.05415653e-04, 1.01213064e-03, 9.71783977e-03, 9.33043882e-02, 8.95848095e-01],
               [1.97478158e-07, 2.55332725e-06, 3.30137336e-05, 4.26855026e-04, 5.51909395e-03, 7.13600516e-02, 9.22658265e-01],
               [3.38271953e-08, 5.88989565e-07, 1.02553395e-05, 1.78562390e-04, 3.10907187e-03, 5.41342832e-02, 9.42567229e-01],
               [5.76100057e-09, 1.35080853e-07, 3.16730916e-06, 7.42652192e-05, 1.74133084e-03, 4.08297926e-02, 9.57351327e-01],
               [3.20762188e-29, 3.20762188e-29, 2.00000003e-01, 2.00000003e-01, 2.00000003e-01, 2.00000003e-01, 2.00000003e-01],]],
            ]),
            rtol:($srvLvl==Service::LV_BASIC)?1e+0:1e-4,
            //debug:true,
        ));

        //echo "scores: ".$mo->toString($K->slice($scores,[-1,-1,-1],[1,1,1]),format:'%12.7e',indent:true)."\n";
        //echo "scores: ".$mo->shapeToString($K->slice($scores,[-1,-1,-1],[1,1,1])->shape())."\n";
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($K->slice($scores,[0,-1],[2,1])),
            $mo->array([
             [[[7.85065666e-02, 9.38552469e-02, 1.12204731e-01, 1.34141684e-01, 1.60367504e-01, 1.91720679e-01, 2.29203656e-01],
               [2.25635562e-02, 3.63257453e-02, 5.84818795e-02, 9.41517130e-02, 1.51577622e-01, 2.44029343e-01, 3.92870098e-01],
               [5.21136262e-03, 1.12982830e-02, 2.44947895e-02, 5.31049334e-02, 1.15132004e-01, 2.49607325e-01, 5.41151285e-01],
               [1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01],
               [1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01],
               [1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01],]],
             
             [[[6.54485575e-06, 4.66635938e-05, 3.32703930e-04, 2.37211213e-03, 1.69127118e-02, 1.20584704e-01, 8.59744608e-01],
               [1.14350473e-06, 1.09792109e-05, 1.05415653e-04, 1.01213064e-03, 9.71783977e-03, 9.33043882e-02, 8.95848095e-01],
               [1.97478158e-07, 2.55332725e-06, 3.30137336e-05, 4.26855026e-04, 5.51909395e-03, 7.13600516e-02, 9.22658265e-01],
               [3.38271953e-08, 5.88989565e-07, 1.02553395e-05, 1.78562390e-04, 3.10907187e-03, 5.41342832e-02, 9.42567229e-01],
               [5.76100057e-09, 1.35080853e-07, 3.16730916e-06, 7.42652192e-05, 1.74133084e-03, 4.08297926e-02, 9.57351327e-01],
               [3.20762188e-29, 3.20762188e-29, 2.00000003e-01, 2.00000003e-01, 2.00000003e-01, 2.00000003e-01, 2.00000003e-01],]],
            ]),
            rtol:($srvLvl==Service::LV_BASIC)?1e+0:1e-4,
            //debug:true,
        ));
        //
        // backward
        //
        // 
        $dResultValiable = $K->ones($resultValiable->shape());
        [$dOutputs,$dSalt] = $resultValiable->creator()->backward([$dResultValiable]);
        $this->assertEquals($outputsVariable->shape(),$dOutputs->shape());
        $this->assertEquals($resultValiable->shape(),$dSalt->shape());
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dOutputs),$K->ndarray($salt)
        ));
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dSalt),$K->ndarray($outputs)
        ));

        $copydOutputs = $K->copy($dOutputs);
        $dInputs = $outputsVariable->creator()->backward([$dOutputs]);
        // 2 batch
        $this->assertCount(2,$dInputs);
        //echo "dQuery: ".$mo->shapeToString($dInputs[0]->shape())."\n";
        //echo "dValue: ".$mo->shapeToString($dInputs[1]->shape())."\n";
        $this->assertEquals($full_query_shape,$dInputs[0]->shape());
        $this->assertEquals($full_value_shape,$dInputs[1]->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());

        //echo "dQuery: ".$mo->toString($dInputs[0],format:'%12.7e',indent:true)."\n";
        //echo "dValue: ".$mo->toString($dInputs[1],format:'%12.7e',indent:true)."\n";

        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dInputs[0]),
            $mo->array([
               [[ 302.03033,   302.03033,   302.03033,   302.03033,   302.03033 ],
                [ 699.38666,   699.38666,   699.38666,   699.38666,   699.38666 ],
                [ 670.80566,   670.80566,   670.80566,   670.80566,   670.80566 ],
                [2775.5093 ,  2775.5093 ,  2775.5093 ,  2775.5093 ,  2775.5093  ],
                [3591.8362 ,  3591.8362 ,  3591.8362 ,  3591.8362 ,  3591.8362  ],
                [4408.163  ,  4408.163  ,  4408.163  ,  4408.163  ,  4408.163   ],],
              
               [[ 247.77393 ,  247.77393 ,  247.77393 ,  247.77393 ,  247.77393 ],
                [ 195.96411 ,  195.96411 ,  195.96411 ,  195.96411 ,  195.96411 ],
                [ 155.72778 ,  155.72778 ,  155.72778 ,  155.72778 ,  155.72778 ],
                [ 123.993286,  123.993286,  123.993286,  123.993286,  123.993286],
                [  98.746216,   98.746216,   98.746216,   98.746216,   98.746216],
                [4653.039   , 4653.039   , 4653.039   , 4653.039   , 4653.039   ],],
            ]),
            rtol:($srvLvl==Service::LV_BASIC)?1e+1:1e-4,
            //debug:true,
        ));
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dInputs[1]),
            $mo->array([
              [[-4.9432705e+03, -4.9432705e+03, -4.9432705e+03, -4.9432705e+03, -4.9432705e+03],
               [-2.8203491e+03, -2.8203491e+03, -2.8203491e+03, -2.8203491e+03, -2.8203491e+03],
               [-6.8445068e+02, -6.8445068e+02, -6.8445068e+02, -6.8445068e+02, -6.8445068e+02],
               [ 1.5006438e+03,  1.5006438e+03,  1.5006438e+03,  1.5006438e+03,  1.5006438e+03],
               [ 3.8357522e+03,  3.8357522e+03,  3.8357522e+03,  3.8357522e+03,  3.8357522e+03],
               [ 6.5880391e+03,  6.5880391e+03,  6.5880391e+03,  6.5880391e+03,  6.5880391e+03],
               [ 1.0443629e+04,  1.0443629e+04,  1.0443629e+04,  1.0443629e+04,  1.0443629e+04],],
             
              [[-4.5769331e-01, -4.5769331e-01, -4.5769331e-01, -4.5769331e-01, -4.5769331e-01],
               [-2.9572980e+00, -2.9572980e+00, -2.9572980e+00, -2.9572980e+00, -2.9572980e+00],
               [-1.0789374e+04, -1.0789374e+04, -1.0789374e+04, -1.0789374e+04, -1.0789374e+04],
               [-4.5953872e+03, -4.5953872e+03, -4.5953872e+03, -4.5953872e+03, -4.5953872e+03],
               [ 1.0847009e+03,  1.0847009e+03,  1.0847009e+03,  1.0847009e+03,  1.0847009e+03],
               [ 5.1574980e+03,  5.1574980e+03,  5.1574980e+03,  5.1574980e+03,  5.1574980e+03],
               [ 5.1865906e+04,  5.1865906e+04,  5.1865906e+04,  5.1865906e+04,  5.1865906e+04],],
            ]),
            rtol:($srvLvl==Service::LV_BASIC)?1e+0:1e-4,
            //debug:true,
        ));

    }

    public function testMaskValueOnly()
    {
        $num_heads = 8;
        $key_dim = 4;
        #$full_query_shape = [2, 6, 16];
        #$full_value_shape = [2, 7, 16];
        $full_query_shape = [2, 6, 5];
        $full_value_shape = [2, 7, 5];
        #$full_query_shape = [2, 3, 6, 5];
        #$full_value_shape = [2, 3, 7, 5];
        //$query_mask = [
        //    [True,True,True,False,False,False],
        //    [True,True,True,True,True,False],
        //];
        $value_mask = [
            [True,True,True,True,False,False,False],
            [True,True,True,True,True,True,False],
        ];
        
        $tmp = $full_query_shape;
        $tSeq = array_splice($tmp,1,-1);
        [$batches,$dim] = $tmp;
        $tmp = $full_value_shape;
        $sSeq = array_splice($tmp,1,-1);
        [$batches,$dim] = $tmp;

        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $la = $K->primaryLA();
        $layer = new MultiHeadAttention(
            $K,
            $num_heads, // num_heads
            $key_dim,   // key_dim
            kernel_initializer:'ones',
            bias_initializer:'zeros',
        );
        $salt_q = $mo->la()->range(array_product($full_query_shape),dtype:NDArray::float32)
                ->reshape($full_query_shape);
        $salt_v = $mo->la()->range(array_product($full_value_shape),dtype:NDArray::float32)
                ->reshape($full_value_shape);
        $salt_q = $K->array($salt_q);
        $salt_v = $K->array($salt_v);
        //$query = $g->Variable($la->randomNormal($full_query_shape,mean:0,scale:1));
        //$value = $g->Variable($la->randomNormal($full_value_shape,mean:0,scale:1));
        //echo "query: ".$mo->toString($dInputs[0],format:'%12.7e',indent:true)."\n";
        $query = $g->Variable($K->scale(1/array_product($full_query_shape),$K->increment($salt_q,1)));
        $value = $g->Variable($K->scale(1/array_product($full_value_shape),$K->increment($salt_v,1)));
        $query_mask = null;
        //$query_mask = $g->Variable($K->array($query_mask,dtype:NDArray::bool));
        $value_mask = $g->Variable($K->array($value_mask,dtype:NDArray::bool));
        //$query = $g->Variable($K->increment(
        //        $K->scale(0.5, $K->array($mo->la()->range(array_product($full_query_shape),dtype:NDArray::float32)
        //        ->reshape($full_query_shape))),
        //    1,
        //));
        //$value = $g->Variable($K->increment(
        //        $K->scale(0.2, $K->array($mo->la()->range(array_product($full_value_shape),dtype:NDArray::float32)
        //        ->reshape($full_value_shape))),
        //    1,
        //));
        //echo 'query:'.$mo->toString($query,format:'%12.7f',indent:true)."\n";
        //echo 'value:'.$mo->toString($value,format:'%12.7f',indent:true)."\n";
        //echo 'query:'.$mo->shapeToString($query->shape())."\n";
        //echo 'value:'.$mo->shapeToString($value->shape())."\n";
        //echo 'query_mask:'.$mo->shapeToString($query_mask->shape())."\n";
        //echo 'value_mask:'.$mo->shapeToString($value_mask->shape())."\n";
        $inputs = [
            $query,
            $value,
        ];

        $layer->build($inputs,
        );

        //
        // forward
        //
        //  batch size 2
        //echo "query: ".$mo->toString($query,format:'%12.7f',indent:true)."\n";
        //////////////echo "key: ".$mo->toString($key,indent:true)."\n";
        //echo "value: ".$mo->toString($value,format:'%12.7f',indent:true)."\n";

        #$query_mask = null;
        #$value_mask = null;
        $salt = $g->Variable($salt_q);
        $copyInputs = [$K->copy($query),$K->copy($value)];
        [$outputsVariable,$scores,$resultValiable] = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$layer,$inputs,$salt,$query_mask,$value_mask) {
                [$outputsVariable,$scores] = $layer->forward(
                    $inputs,
                    training:true,
                    returnAttentionScores:true,
                    mask:[$query_mask,$value_mask],
                );
                $resultValiable = $g->mul($outputsVariable,$salt);
                return [$outputsVariable,$scores,$resultValiable];
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        //echo 'query:'.$mo->toString($query,format:'%12.7f',indent:true)."\n";
        //echo 'value:'.$mo->toString($value,format:'%12.7f',indent:true)."\n";
        //echo 'outputs:'.$mo->toString($outputs,format:'%12.7f',indent:true)."\n";
        //echo 'scores:'.$mo->toString($scores,format:'%12.7e',indent:true)."\n";
        //echo 'kernel:'.$mo->toString($layer->getParams()[0],format:'%14.7f',indent:true)."\n";
        $this->assertEquals(array_merge([$batches, $num_heads], $tSeq, $sSeq),$scores->shape());
        $this->assertEquals(array_merge([$batches], $tSeq, [$dim]),$outputs->shape());
        $this->assertEquals($copyInputs[0]->toArray(),$inputs[0]->toArray());
        $this->assertEquals($copyInputs[1]->toArray(),$inputs[1]->toArray());
        //$this->assertTrue($mo->la()->isclose(
        //    $K->fill([2,8,6,7], 0.14285715),
        //    $K->ndarray($scores)));
        //$this->assertTrue($mo->la()->isclose(
        //    //$K->mul($salt,$K->fill($full_query_shape,512)),
        //    $K->fill($full_query_shape,512),
        //    $K->ndarray($outputs)
        //));
        
        //echo "outputs: ".$mo->toString($outputs,format:'%12.7f',indent:true)."\n";
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($outputs),
            $mo->array([
               [[ 26.528252,  26.528252,  26.528252,  26.528252,  26.528252],
                [ 30.400398,  30.400398,  30.400398,  30.400398,  30.400398],
                [ 33.52552 ,  33.52552 ,  33.52552 ,  33.52552 ,  33.52552 ],
                [ 35.827053,  35.827053,  35.827053,  35.827053,  35.827053],
                [ 37.436584,  37.436584,  37.436584,  37.436584,  37.436584],
                [ 38.539886,  38.539886,  38.539886,  38.539886,  38.539886],],
              
               [[142.1361 ,  142.1361 ,  142.1361 ,  142.1361 ,  142.1361  ],
                [142.67139,  142.67139,  142.67139,  142.67139,  142.67139 ],
                [143.042  ,  143.042  ,  143.042  ,  143.042  ,  143.042   ],
                [143.30368,  143.30368,  143.30368,  143.30368,  143.30368 ],
                [143.49086,  143.49086,  143.49086,  143.49086,  143.49086 ],
                [143.62624,  143.62624,  143.62624,  143.62624,  143.62624 ],],
            ]),
            //debug:true,
        ));


        //$score0_org = $mo->array([
        //     [[[1.87497035e-01, 2.24154279e-01, 2.67978311e-01, 3.20370287e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
        //       [1.06671929e-01, 1.71734333e-01, 2.76480168e-01, 4.45113599e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
        //       [5.53755946e-02, 1.20054819e-01, 2.60280043e-01, 5.64289570e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
        //       [2.67889500e-02, 7.82116279e-02, 2.28342533e-01, 6.66656792e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
        //       [1.23210493e-02, 4.84414771e-02, 1.90452754e-01, 7.48784721e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
        //       [5.47227450e-03, 2.89729107e-02, 1.53396845e-01, 8.12157929e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],]],
//
        //     [[[4.66638303e-05, 3.32704338e-04, 2.37212842e-03, 1.69127975e-02, 1.20585077e-01, 8.59750569e-01, 0.00000000e+00],
        //       [1.09792027e-05, 1.05415376e-04, 1.01213378e-03, 9.71783325e-03, 9.33045000e-02, 8.95849168e-01, 0.00000000e+00],
        //       [2.55331861e-06, 3.30135626e-05, 4.26855200e-04, 5.51907532e-03, 7.13598132e-02, 9.22658682e-01, 0.00000000e+00],
        //       [5.88987461e-07, 1.02552840e-05, 1.78562434e-04, 3.10906675e-03, 5.41340895e-02, 9.42567468e-01, 0.00000000e+00],
        //       [1.35080356e-07, 3.16729142e-06, 7.42650882e-05, 1.74132455e-03, 4.08296399e-02, 9.57351446e-01, 0.00000000e+00],
        //       [3.08513251e-08, 9.74145678e-07, 3.07591836e-05, 9.71232483e-04, 3.06671392e-02, 9.68329906e-01, 0.00000000e+00],]],
        //]);
        //$score0_cst = $mo->array([
        // [[[1.87497050e-01, 2.24154294e-01, 2.67978340e-01, 3.20370287e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
        //   [1.06671929e-01, 1.71734318e-01, 2.76480168e-01, 4.45113599e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
        //   [5.53755946e-02, 1.20054819e-01, 2.60280043e-01, 5.64289570e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
        //   [2.67889518e-02, 7.82116279e-02, 2.28342533e-01, 6.66656792e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
        //   [1.23210484e-02, 4.84414771e-02, 1.90452740e-01, 7.48784721e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
        //   [5.47227450e-03, 2.89729126e-02, 1.53396845e-01, 8.12157929e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],]],
        // 
        // [[[4.66638303e-05, 3.32704338e-04, 2.37212866e-03, 1.69127975e-02, 1.20585077e-01, 8.59750569e-01, 0.00000000e+00],
        //   [1.09792027e-05, 1.05415376e-04, 1.01213378e-03, 9.71783325e-03, 9.33045000e-02, 8.95849168e-01, 0.00000000e+00],
        //   [2.55331861e-06, 3.30135626e-05, 4.26855200e-04, 5.51907532e-03, 7.13598132e-02, 9.22658682e-01, 0.00000000e+00],
        //   [5.88987461e-07, 1.02552831e-05, 1.78562434e-04, 3.10906675e-03, 5.41340895e-02, 9.42567468e-01, 0.00000000e+00],
        //   [1.35080356e-07, 3.16729142e-06, 7.42650882e-05, 1.74132443e-03, 4.08296399e-02, 9.57351446e-01, 0.00000000e+00],
        //   [3.08513251e-08, 9.74145678e-07, 3.07591836e-05, 9.71232483e-04, 3.06671374e-02, 9.68329906e-01, 0.00000000e+00],]],
        //]);
        ////echo "scores0\n";
        ////echo "scores0: ".$mo->toString($K->slice($scores,[0,0],[2,1]),format:'%12.7e',indent:true)."\n";
        //$this->assertTrue($mo->la()->isclose(
        //    //$K->ndarray($K->slice($scores,[0,0],[2,1])),
        //    $score0_org,
        //    $score0_cst,
        //    debug:true,
        //));

        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($K->sum($scores,axis:1)),
            $mo->array([
             [[1.4999763e+00, 1.7932341e+00, 2.1438265e+00, 2.5629623e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
              [8.5337543e-01, 1.3738747e+00, 2.2118413e+00, 3.5609090e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
              [4.4300479e-01, 9.6043861e-01, 2.0822403e+00, 4.5143166e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
              [2.1431160e-01, 6.2569302e-01, 1.8267403e+00, 5.3332543e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
              [9.8568387e-02, 3.8753179e-01, 1.5236220e+00, 5.9902773e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
              [4.3778196e-02, 2.3178329e-01, 1.2271748e+00, 6.4972639e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],],
            
             [[3.7331064e-04, 2.6616345e-03, 1.8977029e-02, 1.3530238e-01, 9.6468061e-01, 6.8780050e+00, 0.0000000e+00],
              [8.7833621e-05, 8.4332295e-04, 8.0970703e-03, 7.7742666e-02, 7.4643606e-01, 7.1667933e+00, 0.0000000e+00],
              [2.0426551e-05, 2.6410850e-04, 3.4148416e-03, 4.4152603e-02, 5.7087851e-01, 7.3812704e+00, 0.0000000e+00],
              [4.7118997e-06, 8.2042272e-05, 1.4284996e-03, 2.4872534e-02, 4.3307275e-01, 7.5405393e+00, 0.0000000e+00],
              [1.0806428e-06, 2.5338331e-05, 5.9412071e-04, 1.3930596e-02, 3.2663712e-01, 7.6588125e+00, 0.0000000e+00],
              [2.4681060e-07, 7.7931654e-06, 2.4607347e-04, 7.7698599e-03, 2.4533711e-01, 7.7466393e+00, 0.0000000e+00],],
            ]),
            //debug:true,
        ));

        //
        // backward
        //
        // 
        $dResultValiable = $K->ones($resultValiable->shape());
        [$dOutputs,$dSalt] = $resultValiable->creator()->backward([$dResultValiable]);
        $this->assertEquals($outputsVariable->shape(),$dOutputs->shape());
        $this->assertEquals($resultValiable->shape(),$dSalt->shape());
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dOutputs),$K->ndarray($salt)
        ));
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dSalt),$K->ndarray($outputs),
            //debug:true,
        ));

        $copydOutputs = $K->copy($dOutputs);
        $dInputs = $outputsVariable->creator()->backward([$dOutputs]);
        // 2 batch
        $this->assertCount(2,$dInputs);
        //echo "dQuery: ".$mo->shapeToString($dInputs[0]->shape())."\n";
        //echo "dValue: ".$mo->shapeToString($dInputs[1]->shape())."\n";
        $this->assertEquals($full_query_shape,$dInputs[0]->shape());
        $this->assertEquals($full_value_shape,$dInputs[1]->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());

        //echo "dQuery: ".$mo->toString($dInputs[0],format:'%12.7f',indent:true)."\n";
        //echo "dValue: ".$mo->toString($dInputs[1],format:'%12.7e',indent:true)."\n";

        $dq_org = $mo->array([ // original tf.nn.softmax
               [[ 99.330315,  99.330315,  99.330315,  99.330315,  99.330315],
                [297.12323 , 297.12323 , 297.12323 , 297.12323 , 297.12323 ],
                [389.20917 , 389.20917 , 389.20917 , 389.20917 , 389.20917 ],
                [392.6947  , 392.6947  , 392.6947  , 392.6947  , 392.6947  ],
                [350.1486  , 350.1486  , 350.1486  , 350.1486  , 350.1486  ],
                [293.8765  , 293.8765  , 293.8765  , 293.8765  , 293.8765  ],],
              
               [[247.50195 , 247.50195 , 247.50195 , 247.50195 , 247.50195 ],
                [195.90976 , 195.90976 , 195.90976 , 195.90976 , 195.90976 ],
                [155.72949 , 155.72949 , 155.72949 , 155.72949 , 155.72949 ],
                [124.00012 , 124.00012 , 124.00012 , 124.00012 , 124.00012 ],
                [ 98.75073 ,  98.75073 ,  98.75073 ,  98.75073 ,  98.75073 ],
                [ 78.574585,  78.574585,  78.574585,  78.574585,  78.574585],],
        ]);
        
        $dq_cst = $mo->array([ // custom softmax
               [[ 99.330215,  99.330215,  99.330215,  99.330215,  99.330215],
                [297.12323 , 297.12323 , 297.12323 , 297.12323 , 297.12323 ],
                [389.209   , 389.209   , 389.209   , 389.209   , 389.209   ],
                [392.6937  , 392.6937  , 392.6937  , 392.6937  , 392.6937  ],
                [350.14862 , 350.14862 , 350.14862 , 350.14862 , 350.14862 ],
                [293.87515 , 293.87515 , 293.87515 , 293.87515 , 293.87515 ],],
              
               [[247.47974 , 247.47974 , 247.47974 , 247.47974 , 247.47974 ],
                [195.92099 , 195.92099 , 195.92099 , 195.92099 , 195.92099 ],
                [155.73157 , 155.73157 , 155.73157 , 155.73157 , 155.73157 ],
                [124.01001 , 124.01001 , 124.01001 , 124.01001 , 124.01001 ],
                [ 98.76343 ,  98.76343 ,  98.76343 ,  98.76343 ,  98.76343 ],
                [ 78.579895,  78.579895,  78.579895,  78.579895,  78.579895],],
        ]);
        //echo "diff org-cst: ".$mo->toString($K->sub($dq_org,$dq_cst),format:'%12.7f',indent:true)."\n";
        //echo "diff-org : ".$mo->toString($K->sum($K->abs($K->sub($dInputs[0],$dq_cst)),axis:-1),format:'%12.7f',indent:true)."\n";
        //echo "diff-cst : ".$mo->toString($K->sum($K->abs($K->sub($dInputs[0],$dq_org)),axis:-1),format:'%12.7f',indent:true)."\n";

        //echo "dQuery: ".$mo->toString($K->sub($dInputs[0],$dq_cst),format:'%12.7f',indent:true)."\n";
        //echo "dQuery: ".$mo->toString($K->sub($dInputs[0],$dq_org),format:'%12.7f',indent:true)."\n";
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dInputs[0]),
            $dq_org,
            //atol:1e-1,
            rtol:1e-3,
            //debug:true,
        ));
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dInputs[1]),
            $mo->array([
              [[-3.67950104e+02, -3.67950104e+02, -3.67950104e+02, -3.67950104e+02, -3.67950104e+02],
               [-4.51909058e+02, -4.51909058e+02, -4.51909058e+02, -4.51909058e+02, -4.51909058e+02],
               [ 8.86771118e+02,  8.86771118e+02,  8.86771118e+02,  8.86771118e+02,  8.86771118e+02],
               [ 1.38530908e+04,  1.38530908e+04,  1.38530908e+04,  1.38530908e+04,  1.38530908e+04],
               [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
               [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
               [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],],
             
              [[-2.96187091e+00, -2.96187091e+00, -2.96187091e+00, -2.96187091e+00, -2.96187091e+00],
               [-1.92011375e+01, -1.92011375e+01, -1.92011375e+01, -1.92011375e+01, -1.92011375e+01],
               [-1.24836975e+02, -1.24836975e+02, -1.24836975e+02, -1.24836975e+02, -1.24836975e+02],
               [-7.90601074e+02, -7.90601074e+02, -7.90601074e+02, -7.90601074e+02, -7.90601074e+02],
               [-3.61800879e+03, -3.61800879e+03, -3.61800879e+03, -3.61800879e+03, -3.61800879e+03],
               [ 4.72755898e+04,  4.72755898e+04,  4.72755898e+04,  4.72755898e+04,  4.72755898e+04],
               [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],],
            ]),
            //debug:true,
        ));

    }

    public function testMaskKeyOnly()
    {
        $num_heads = 8;
        $key_dim = 4;
        #$full_query_shape = [2, 6, 16];
        #$full_value_shape = [2, 7, 16];
        $full_query_shape = [2, 6, 5];
        $full_value_shape = [2, 7, 5];
        #$full_query_shape = [2, 3, 6, 5];
        #$full_value_shape = [2, 3, 7, 5];
        //$query_mask = [
        //    [True,True,True,False,False,False],
        //    [True,True,True,True,True,False],
        //];
        $value_mask = [
            [True,True,True,True,False,False,False],
            [True,True,True,True,True,True,False],
        ];
        
        $tmp = $full_query_shape;
        $tSeq = array_splice($tmp,1,-1);
        [$batches,$dim] = $tmp;
        $tmp = $full_value_shape;
        $sSeq = array_splice($tmp,1,-1);
        [$batches,$dim] = $tmp;

        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $la = $K->primaryLA();
        $layer = new MultiHeadAttention(
            $K,
            $num_heads, // num_heads
            $key_dim,   // key_dim
            kernel_initializer:'ones',
            bias_initializer:'zeros',
        );
        $salt_q = $mo->la()->range(array_product($full_query_shape),dtype:NDArray::float32)
                ->reshape($full_query_shape);
        $salt_v = $mo->la()->range(array_product($full_value_shape),dtype:NDArray::float32)
                ->reshape($full_value_shape);
        $salt_q = $K->array($salt_q);
        $salt_v = $K->array($salt_v);
        //$query = $g->Variable($la->randomNormal($full_query_shape,mean:0,scale:1));
        //$value = $g->Variable($la->randomNormal($full_value_shape,mean:0,scale:1));
        //echo "query: ".$mo->toString($dInputs[0],format:'%12.7e',indent:true)."\n";
        $query = $g->Variable($K->scale(1/array_product($full_query_shape),$K->increment($salt_q,1)));
        $value = $g->Variable($K->scale(1/array_product($full_value_shape),$K->increment($salt_v,1)));
        $key = $g->Variable($K->scale(1/array_product($full_value_shape),$K->increment($salt_v,1)));
        //$query_mask = $g->Variable($K->array($query_mask,dtype:NDArray::bool));
        //$value_mask = $g->Variable($K->array($value_mask,dtype:NDArray::bool));
        $key_mask = $g->Variable($K->array($value_mask,dtype:NDArray::bool));
        $query_mask = null;
        $value_mask = null;
        //$query = $g->Variable($K->increment(
        //        $K->scale(0.5, $K->array($mo->la()->range(array_product($full_query_shape),dtype:NDArray::float32)
        //        ->reshape($full_query_shape))),
        //    1,
        //));
        //$value = $g->Variable($K->increment(
        //        $K->scale(0.2, $K->array($mo->la()->range(array_product($full_value_shape),dtype:NDArray::float32)
        //        ->reshape($full_value_shape))),
        //    1,
        //));
        //echo 'query:'.$mo->toString($query,format:'%12.7f',indent:true)."\n";
        //echo 'value:'.$mo->toString($value,format:'%12.7f',indent:true)."\n";
        //echo 'query:'.$mo->shapeToString($query->shape())."\n";
        //echo 'value:'.$mo->shapeToString($value->shape())."\n";
        //echo 'query_mask:'.$mo->shapeToString($query_mask->shape())."\n";
        //echo 'value_mask:'.$mo->shapeToString($value_mask->shape())."\n";
        $inputs = [
            $query,
            $value,
            $key,
        ];

        $layer->build($inputs,
        );

        //
        // forward
        //
        //  batch size 2
        //echo "query: ".$mo->toString($query,format:'%12.7f',indent:true)."\n";
        //////////////echo "key: ".$mo->toString($key,indent:true)."\n";
        //echo "value: ".$mo->toString($value,format:'%12.7f',indent:true)."\n";

        #$query_mask = null;
        #$value_mask = null;
        $salt = $g->Variable($salt_q);
        $copyInputs = [$K->copy($query),$K->copy($value)];
        [$outputsVariable,$scores,$resultValiable] = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$layer,$inputs,$salt,$query_mask,$value_mask,$key_mask) {
                [$outputsVariable,$scores] = $layer->forward(
                    $inputs,
                    training:true,
                    returnAttentionScores:true,
                    mask:[$query_mask,$value_mask,$key_mask],
                );
                $resultValiable = $g->mul($outputsVariable,$salt);
                return [$outputsVariable,$scores,$resultValiable];
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        //echo 'query:'.$mo->toString($query,format:'%12.7f',indent:true)."\n";
        //echo 'value:'.$mo->toString($value,format:'%12.7f',indent:true)."\n";
        //echo 'outputs:'.$mo->toString($outputs,format:'%12.7f',indent:true)."\n";
        //echo 'scores:'.$mo->toString($scores,format:'%12.7e',indent:true)."\n";
        //echo 'kernel:'.$mo->toString($layer->getParams()[0],format:'%14.7f',indent:true)."\n";
        $this->assertEquals(array_merge([$batches, $num_heads], $tSeq, $sSeq),$scores->shape());
        $this->assertEquals(array_merge([$batches], $tSeq, [$dim]),$outputs->shape());
        $this->assertEquals($copyInputs[0]->toArray(),$inputs[0]->toArray());
        $this->assertEquals($copyInputs[1]->toArray(),$inputs[1]->toArray());
        //$this->assertTrue($mo->la()->isclose(
        //    $K->fill([2,8,6,7], 0.14285715),
        //    $K->ndarray($scores)));
        //$this->assertTrue($mo->la()->isclose(
        //    //$K->mul($salt,$K->fill($full_query_shape,512)),
        //    $K->fill($full_query_shape,512),
        //    $K->ndarray($outputs)
        //));
        
        //echo "outputs: ".$mo->toString($outputs,format:'%12.7f',indent:true)."\n";
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($outputs),
            $mo->array([
               [[ 26.528252,  26.528252,  26.528252,  26.528252,  26.528252],
                [ 30.400398,  30.400398,  30.400398,  30.400398,  30.400398],
                [ 33.52552 ,  33.52552 ,  33.52552 ,  33.52552 ,  33.52552 ],
                [ 35.827053,  35.827053,  35.827053,  35.827053,  35.827053],
                [ 37.436584,  37.436584,  37.436584,  37.436584,  37.436584],
                [ 38.539886,  38.539886,  38.539886,  38.539886,  38.539886],],
              
               [[142.1361 ,  142.1361 ,  142.1361 ,  142.1361 ,  142.1361  ],
                [142.67139,  142.67139,  142.67139,  142.67139,  142.67139 ],
                [143.042  ,  143.042  ,  143.042  ,  143.042  ,  143.042   ],
                [143.30368,  143.30368,  143.30368,  143.30368,  143.30368 ],
                [143.49086,  143.49086,  143.49086,  143.49086,  143.49086 ],
                [143.62624,  143.62624,  143.62624,  143.62624,  143.62624 ],],
            ]),
            //debug:true,
        ));


        //$score0_org = $mo->array([
        //     [[[1.87497035e-01, 2.24154279e-01, 2.67978311e-01, 3.20370287e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
        //       [1.06671929e-01, 1.71734333e-01, 2.76480168e-01, 4.45113599e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
        //       [5.53755946e-02, 1.20054819e-01, 2.60280043e-01, 5.64289570e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
        //       [2.67889500e-02, 7.82116279e-02, 2.28342533e-01, 6.66656792e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
        //       [1.23210493e-02, 4.84414771e-02, 1.90452754e-01, 7.48784721e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
        //       [5.47227450e-03, 2.89729107e-02, 1.53396845e-01, 8.12157929e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],]],
//
        //     [[[4.66638303e-05, 3.32704338e-04, 2.37212842e-03, 1.69127975e-02, 1.20585077e-01, 8.59750569e-01, 0.00000000e+00],
        //       [1.09792027e-05, 1.05415376e-04, 1.01213378e-03, 9.71783325e-03, 9.33045000e-02, 8.95849168e-01, 0.00000000e+00],
        //       [2.55331861e-06, 3.30135626e-05, 4.26855200e-04, 5.51907532e-03, 7.13598132e-02, 9.22658682e-01, 0.00000000e+00],
        //       [5.88987461e-07, 1.02552840e-05, 1.78562434e-04, 3.10906675e-03, 5.41340895e-02, 9.42567468e-01, 0.00000000e+00],
        //       [1.35080356e-07, 3.16729142e-06, 7.42650882e-05, 1.74132455e-03, 4.08296399e-02, 9.57351446e-01, 0.00000000e+00],
        //       [3.08513251e-08, 9.74145678e-07, 3.07591836e-05, 9.71232483e-04, 3.06671392e-02, 9.68329906e-01, 0.00000000e+00],]],
        //]);
        //$score0_cst = $mo->array([
        // [[[1.87497050e-01, 2.24154294e-01, 2.67978340e-01, 3.20370287e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
        //   [1.06671929e-01, 1.71734318e-01, 2.76480168e-01, 4.45113599e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
        //   [5.53755946e-02, 1.20054819e-01, 2.60280043e-01, 5.64289570e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
        //   [2.67889518e-02, 7.82116279e-02, 2.28342533e-01, 6.66656792e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
        //   [1.23210484e-02, 4.84414771e-02, 1.90452740e-01, 7.48784721e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
        //   [5.47227450e-03, 2.89729126e-02, 1.53396845e-01, 8.12157929e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],]],
        // 
        // [[[4.66638303e-05, 3.32704338e-04, 2.37212866e-03, 1.69127975e-02, 1.20585077e-01, 8.59750569e-01, 0.00000000e+00],
        //   [1.09792027e-05, 1.05415376e-04, 1.01213378e-03, 9.71783325e-03, 9.33045000e-02, 8.95849168e-01, 0.00000000e+00],
        //   [2.55331861e-06, 3.30135626e-05, 4.26855200e-04, 5.51907532e-03, 7.13598132e-02, 9.22658682e-01, 0.00000000e+00],
        //   [5.88987461e-07, 1.02552831e-05, 1.78562434e-04, 3.10906675e-03, 5.41340895e-02, 9.42567468e-01, 0.00000000e+00],
        //   [1.35080356e-07, 3.16729142e-06, 7.42650882e-05, 1.74132443e-03, 4.08296399e-02, 9.57351446e-01, 0.00000000e+00],
        //   [3.08513251e-08, 9.74145678e-07, 3.07591836e-05, 9.71232483e-04, 3.06671374e-02, 9.68329906e-01, 0.00000000e+00],]],
        //]);
        ////echo "scores0\n";
        ////echo "scores0: ".$mo->toString($K->slice($scores,[0,0],[2,1]),format:'%12.7e',indent:true)."\n";
        //$this->assertTrue($mo->la()->isclose(
        //    //$K->ndarray($K->slice($scores,[0,0],[2,1])),
        //    $score0_org,
        //    $score0_cst,
        //    debug:true,
        //));

        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($K->sum($scores,axis:1)),
            $mo->array([
             [[1.4999763e+00, 1.7932341e+00, 2.1438265e+00, 2.5629623e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
              [8.5337543e-01, 1.3738747e+00, 2.2118413e+00, 3.5609090e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
              [4.4300479e-01, 9.6043861e-01, 2.0822403e+00, 4.5143166e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
              [2.1431160e-01, 6.2569302e-01, 1.8267403e+00, 5.3332543e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
              [9.8568387e-02, 3.8753179e-01, 1.5236220e+00, 5.9902773e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
              [4.3778196e-02, 2.3178329e-01, 1.2271748e+00, 6.4972639e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],],
            
             [[3.7331064e-04, 2.6616345e-03, 1.8977029e-02, 1.3530238e-01, 9.6468061e-01, 6.8780050e+00, 0.0000000e+00],
              [8.7833621e-05, 8.4332295e-04, 8.0970703e-03, 7.7742666e-02, 7.4643606e-01, 7.1667933e+00, 0.0000000e+00],
              [2.0426551e-05, 2.6410850e-04, 3.4148416e-03, 4.4152603e-02, 5.7087851e-01, 7.3812704e+00, 0.0000000e+00],
              [4.7118997e-06, 8.2042272e-05, 1.4284996e-03, 2.4872534e-02, 4.3307275e-01, 7.5405393e+00, 0.0000000e+00],
              [1.0806428e-06, 2.5338331e-05, 5.9412071e-04, 1.3930596e-02, 3.2663712e-01, 7.6588125e+00, 0.0000000e+00],
              [2.4681060e-07, 7.7931654e-06, 2.4607347e-04, 7.7698599e-03, 2.4533711e-01, 7.7466393e+00, 0.0000000e+00],],
            ]),
            //debug:true,
        ));

        //
        // backward
        //
        // 
        $dResultValiable = $K->ones($resultValiable->shape());
        [$dOutputs,$dSalt] = $resultValiable->creator()->backward([$dResultValiable]);
        $this->assertEquals($outputsVariable->shape(),$dOutputs->shape());
        $this->assertEquals($resultValiable->shape(),$dSalt->shape());
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dOutputs),$K->ndarray($salt)
        ));
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dSalt),$K->ndarray($outputs),
            //debug:true,
        ));

        $copydOutputs = $K->copy($dOutputs);
        $dInputs = $outputsVariable->creator()->backward([$dOutputs]);
        // 2 batch
        $this->assertCount(3,$dInputs);
        //echo "dQuery: ".$mo->shapeToString($dInputs[0]->shape())."\n";
        //echo "dValue: ".$mo->shapeToString($dInputs[1]->shape())."\n";
        $this->assertEquals($full_query_shape,$dInputs[0]->shape());
        $this->assertEquals($full_value_shape,$dInputs[1]->shape());
        $this->assertEquals($full_value_shape,$dInputs[2]->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());

        //echo "dQuery: ".$mo->toString($dInputs[0],format:'%12.7f',indent:true)."\n";
        //echo "dValue: ".$mo->toString($dInputs[1],format:'%12.7e',indent:true)."\n";

        //echo "dQuery: ".$mo->toString($K->sub($dInputs[0],$dq_cst),format:'%12.7f',indent:true)."\n";
        //echo "dQuery: ".$mo->toString($K->sub($dInputs[0],$dq_org),format:'%12.7f',indent:true)."\n";
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dInputs[0]),
            $mo->array([
                [[ 99.330315,  99.330315,  99.330315,  99.330315,  99.330315],
                [297.12323 , 297.12323 , 297.12323 , 297.12323 , 297.12323 ],
                [389.20917 , 389.20917 , 389.20917 , 389.20917 , 389.20917 ],
                [392.6947  , 392.6947  , 392.6947  , 392.6947  , 392.6947  ],
                [350.1486  , 350.1486  , 350.1486  , 350.1486  , 350.1486  ],
                [293.8765  , 293.8765  , 293.8765  , 293.8765  , 293.8765  ],],
              
               [[247.50195 , 247.50195 , 247.50195 , 247.50195 , 247.50195 ],
                [195.90976 , 195.90976 , 195.90976 , 195.90976 , 195.90976 ],
                [155.72949 , 155.72949 , 155.72949 , 155.72949 , 155.72949 ],
                [124.00012 , 124.00012 , 124.00012 , 124.00012 , 124.00012 ],
                [ 98.75073 ,  98.75073 ,  98.75073 ,  98.75073 ,  98.75073 ],
                [ 78.574585,  78.574585,  78.574585,  78.574585,  78.574585],],
            ]),
            //atol:1e-1,
            rtol:1e-3,
            //debug:true,
        ));
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dInputs[1]),
            $mo->array([
               [[4.2566898e+02, 4.2566898e+02, 4.2566898e+02, 4.2566898e+02, 4.2566898e+02],
                [1.0029895e+03, 1.0029895e+03, 1.0029895e+03, 1.0029895e+03, 1.0029895e+03],
                [2.8493083e+03, 2.8493083e+03, 2.8493083e+03, 2.8493083e+03, 2.8493083e+03],
                [9.6420322e+03, 9.6420322e+03, 9.6420322e+03, 9.6420322e+03, 9.6420322e+03],
                [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
                [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
                [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],],
              
               [[3.2690841e-01, 3.2690841e-01, 3.2690841e-01, 3.2690841e-01, 3.2690841e-01],
                [2.6617124e+00, 2.6617124e+00, 2.6617124e+00, 2.6617124e+00, 2.6617124e+00],
                [2.3246796e+01, 2.3246796e+01, 2.3246796e+01, 2.3246796e+01, 2.3246796e+01],
                [2.2793695e+02, 2.2793695e+02, 2.2793695e+02, 2.2793695e+02, 2.2793695e+02],
                [2.6757720e+03, 2.6757720e+03, 2.6757720e+03, 2.6757720e+03, 2.6757720e+03],
                [3.9790059e+04, 3.9790059e+04, 3.9790059e+04, 3.9790059e+04, 3.9790059e+04],
                [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],],
            ]),
            //debug:true,
        ));
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dInputs[2]),
            $mo->array([
              [[-7.9361908e+02, -7.9361908e+02, -7.9361908e+02, -7.9361908e+02, -7.9361908e+02],
               [-1.4548986e+03, -1.4548986e+03, -1.4548986e+03, -1.4548986e+03, -1.4548986e+03],
               [-1.9625372e+03, -1.9625372e+03, -1.9625372e+03, -1.9625372e+03, -1.9625372e+03],
               [ 4.2110586e+03,  4.2110586e+03,  4.2110586e+03,  4.2110586e+03,  4.2110586e+03],
               [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
               [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
               [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00],],
             
              [[-3.2887793e+00, -3.2887793e+00, -3.2887793e+00, -3.2887793e+00, -3.2887793e+00],
               [-2.1862850e+01, -2.1862850e+01, -2.1862850e+01, -2.1862850e+01, -2.1862850e+01],
               [-1.4808377e+02, -1.4808377e+02, -1.4808377e+02, -1.4808377e+02, -1.4808377e+02],
               [-1.0185380e+03, -1.0185380e+03, -1.0185380e+03, -1.0185380e+03, -1.0185380e+03],
               [-6.2937808e+03, -6.2937808e+03, -6.2937808e+03, -6.2937808e+03, -6.2937808e+03],
               [ 7.4855312e+03,  7.4855312e+03,  7.4855312e+03,  7.4855312e+03,  7.4855312e+03],
               [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00],],
            ]),
            //debug:true,
        ));

    }

    public function testAutoMask()
    {
        $num_heads = 8;
        $key_dim = 4;
        #$full_query_shape = [2, 6, 16];
        #$full_value_shape = [2, 7, 16];
        $full_query_shape = [2, 6, 5];
        $full_value_shape = [2, 7, 5];
        #$full_query_shape = [2, 3, 6, 5];
        #$full_value_shape = [2, 3, 7, 5];
        $query_mask = [
            [True,True,True,False,False,False],
            [True,True,True,True,True,False],
        ];
        $value_mask = [
            [True,True,True,True,False,False,False],
            [True,True,True,True,True,True,False],
        ];
        
        $tmp = $full_query_shape;
        $tSeq = array_splice($tmp,1,-1);
        [$batches,$dim] = $tmp;
        $tmp = $full_value_shape;
        $sSeq = array_splice($tmp,1,-1);
        [$batches,$dim] = $tmp;

        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $la = $K->primaryLA();
        $srvLvl = $K->localMatrixOperator()->service()->serviceLevel();
        $layer = new MultiHeadAttention(
            $K,
            $num_heads, // num_heads
            $key_dim,   // key_dim
            kernel_initializer:'ones',
            bias_initializer:'zeros',
        );

        $salt_q = $mo->la()->range(array_product($full_query_shape),dtype:NDArray::float32)
                ->reshape($full_query_shape);
        $salt_v = $mo->la()->range(array_product($full_value_shape),dtype:NDArray::float32)
                ->reshape($full_value_shape);
        $salt_q = $K->array($salt_q);
        $salt_v = $K->array($salt_v);
        //$query = $g->Variable($la->randomNormal($full_query_shape,mean:0,scale:1));
        //$value = $g->Variable($la->randomNormal($full_value_shape,mean:0,scale:1));
        //echo "query: ".$mo->toString($dInputs[0],format:'%12.7e',indent:true)."\n";
        $query = $K->scale(1/array_product($full_query_shape),$K->increment($salt_q,1));
        $value = $K->scale(1/array_product($full_value_shape),$K->increment($salt_v,1));
        $query_mask = $K->array($query_mask,dtype:NDArray::bool);
        $value_mask = $K->array($value_mask,dtype:NDArray::bool);
        //$key_mask = $K->array($value_mask,dtype:NDArray::bool);
        $query = $g->Variable($this->maskedValue($query,$query_mask));
        $value = $g->Variable($this->maskedValue($value,$value_mask));
        //$key = $g->Variable($this->maskedValue($key,$key_mask));
        //$query = $g->Variable($K->increment(
        //        $K->scale(0.5, $K->array($mo->la()->range(array_product($full_query_shape),dtype:NDArray::float32)
        //        ->reshape($full_query_shape))),
        //    1,
        //));
        //$value = $g->Variable($K->increment(
        //        $K->scale(0.2, $K->array($mo->la()->range(array_product($full_value_shape),dtype:NDArray::float32)
        //        ->reshape($full_value_shape))),
        //    1,
        //));
        //echo 'query:'.$mo->toString($query,format:'%12.7f',indent:true)."\n";
        //echo 'value:'.$mo->toString($value,format:'%12.7f',indent:true)."\n";
        //echo 'query:'.$mo->shapeToString($query->shape())."\n";
        //echo 'value:'.$mo->shapeToString($value->shape())."\n";
        //echo 'query_mask:'.$mo->shapeToString($query_mask->shape())."\n";
        //echo 'value_mask:'.$mo->shapeToString($value_mask->shape())."\n";
        $inputs = [
            $query,
            $value,
        ];

        $layer->build($inputs,
        );

        //
        // forward
        //
        //  batch size 2
        //echo "query: ".$mo->toString($query,format:'%12.7f',indent:true)."\n";
        //////////////echo "key: ".$mo->toString($key,indent:true)."\n";
        //echo "value: ".$mo->toString($value,format:'%12.7f',indent:true)."\n";

        #$query_mask = null;
        #$value_mask = null;
        $salt = $g->Variable($salt_q);
        $copyInputs = [$K->copy($query),$K->copy($value)];
        [$outputsVariable,$scores,$resultValiable] = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$layer,$inputs,$salt) {
                [$outputsVariable,$scores] = $layer->forward(
                    $inputs,
                    training:true,
                    returnAttentionScores:true,
                );
                $resultValiable = $g->mul($outputsVariable,$salt);
                return [$outputsVariable,$scores,$resultValiable];
            }
        );
        $this->assertInstanceof(MaskedNDArray::class,$outputsVariable->value());
        $this->assertEquals(array_merge([$batches], $tSeq),$outputsVariable->value()->mask()->shape());
        $outputs = $K->ndarray($outputsVariable);
        //echo 'query:'.$mo->toString($query,format:'%12.7f',indent:true)."\n";
        //echo 'value:'.$mo->toString($value,format:'%12.7f',indent:true)."\n";
        //echo 'outputs:'.$mo->toString($outputs,format:'%12.7f',indent:true)."\n";
        //echo 'scores:'.$mo->toString($scores,format:'%12.7e',indent:true)."\n";
        //echo 'kernel:'.$mo->toString($layer->getParams()[0],format:'%14.7f',indent:true)."\n";
        $this->assertEquals(array_merge([$batches, $num_heads], $tSeq, $sSeq),$scores->shape());
        $this->assertEquals(array_merge([$batches], $tSeq, [$dim]),$outputs->shape());
        $this->assertEquals($copyInputs[0]->toArray(),$inputs[0]->toArray());
        $this->assertEquals($copyInputs[1]->toArray(),$inputs[1]->toArray());
        //$this->assertTrue($mo->la()->isclose(
        //    $K->fill([2,8,6,7], 0.14285715),
        //    $K->ndarray($scores)));
        //$this->assertTrue($mo->la()->isclose(
        //    //$K->mul($salt,$K->fill($full_query_shape,512)),
        //    $K->fill($full_query_shape,512),
        //    $K->ndarray($outputs)
        //));
        
        //echo "outputs: ".$mo->toString($outputs,format:'%12.7f',indent:true)."\n";
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($outputs),
            $mo->array([
               [[ 26.528252,  26.528252,  26.528252,  26.528252,  26.528252],
                [ 30.400398,  30.400398,  30.400398,  30.400398,  30.400398],
                [ 33.52552 ,  33.52552 ,  33.52552 ,  33.52552 ,  33.52552 ],
                [ 41.14286 ,  41.14286 ,  41.14286 ,  41.14286 ,  41.14286 ],
                [ 41.14286 ,  41.14286 ,  41.14286 ,  41.14286 ,  41.14286 ],
                [ 41.14286 ,  41.14286 ,  41.14286 ,  41.14286 ,  41.14286 ],],
              
               [[142.1361 ,  142.1361 ,  142.1361 ,  142.1361 ,  142.1361  ],
                [142.67139,  142.67139,  142.67139,  142.67139,  142.67139 ],
                [143.042  ,  143.042  ,  143.042  ,  143.042  ,  143.042   ],
                [143.30368,  143.30368,  143.30368,  143.30368,  143.30368 ],
                [143.49086,  143.49086,  143.49086,  143.49086,  143.49086 ],
                [132.57143,  132.57143,  132.57143,  132.57143,  132.57143 ],],
            ]),
            rtol:($srvLvl==Service::LV_BASIC)?1e-1:1e-4,
            //debug:true,
        ));

        //echo "scores0: ".$mo->toString($K->slice($scores,[0,0],[2,1]),format:'%12.7e',indent:true)."\n";
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($K->slice($scores,[0,0],[2,1])),
            $mo->array([
             [[[1.87497035e-01, 2.24154279e-01, 2.67978311e-01, 3.20370287e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
               [1.06671929e-01, 1.71734333e-01, 2.76480168e-01, 4.45113599e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
               [5.53755946e-02, 1.20054819e-01, 2.60280043e-01, 5.64289570e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
               [1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01],
               [1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01],
               [1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01],]],
             
             [[[4.66638303e-05, 3.32704338e-04, 2.37212842e-03, 1.69127975e-02, 1.20585077e-01, 8.59750569e-01, 0.00000000e+00],
               [1.09792027e-05, 1.05415376e-04, 1.01213378e-03, 9.71783325e-03, 9.33045000e-02, 8.95849168e-01, 0.00000000e+00],
               [2.55331861e-06, 3.30135626e-05, 4.26855200e-04, 5.51907532e-03, 7.13598132e-02, 9.22658682e-01, 0.00000000e+00],
               [5.88987461e-07, 1.02552840e-05, 1.78562434e-04, 3.10906675e-03, 5.41340895e-02, 9.42567468e-01, 0.00000000e+00],
               [1.35080356e-07, 3.16729142e-06, 7.42650882e-05, 1.74132455e-03, 4.08296399e-02, 9.57351446e-01, 0.00000000e+00],
               [3.20762188e-29, 3.20762188e-29, 2.00000003e-01, 2.00000003e-01, 2.00000003e-01, 2.00000003e-01, 2.00000003e-01],]],
            ]),
            rtol:($srvLvl==Service::LV_BASIC)?1e+0:1e-4,
            //debug:true,
        ));

        //echo "scores: ".$mo->toString($K->slice($scores,[-1,-1,-1],[1,1,1]),format:'%12.7e',indent:true)."\n";
        //echo "scores: ".$mo->shapeToString($K->slice($scores,[-1,-1,-1],[1,1,1])->shape())."\n";
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($K->slice($scores,[0,-1],[2,1])),
            $mo->array([
             [[[1.87497035e-01, 2.24154279e-01, 2.67978311e-01, 3.20370287e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
               [1.06671929e-01, 1.71734333e-01, 2.76480168e-01, 4.45113599e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
               [5.53755946e-02, 1.20054819e-01, 2.60280043e-01, 5.64289570e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
               [1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01],
               [1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01],
               [1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01, 1.42857149e-01],]],
             
             [[[4.66638303e-05, 3.32704338e-04, 2.37212842e-03, 1.69127975e-02, 1.20585077e-01, 8.59750569e-01, 0.00000000e+00],
               [1.09792027e-05, 1.05415376e-04, 1.01213378e-03, 9.71783325e-03, 9.33045000e-02, 8.95849168e-01, 0.00000000e+00],
               [2.55331861e-06, 3.30135626e-05, 4.26855200e-04, 5.51907532e-03, 7.13598132e-02, 9.22658682e-01, 0.00000000e+00],
               [5.88987461e-07, 1.02552840e-05, 1.78562434e-04, 3.10906675e-03, 5.41340895e-02, 9.42567468e-01, 0.00000000e+00],
               [1.35080356e-07, 3.16729142e-06, 7.42650882e-05, 1.74132455e-03, 4.08296399e-02, 9.57351446e-01, 0.00000000e+00],
               [3.20762188e-29, 3.20762188e-29, 2.00000003e-01, 2.00000003e-01, 2.00000003e-01, 2.00000003e-01, 2.00000003e-01],]],
            ]),
            rtol:($srvLvl==Service::LV_BASIC)?1e+0:1e-4,
            //debug:true,
        ));
        //
        // backward
        //
        // 
        $dResultValiable = $K->ones($resultValiable->shape());
        [$dOutputs,$dSalt] = $resultValiable->creator()->backward([$dResultValiable]);
        $this->assertEquals($outputsVariable->shape(),$dOutputs->shape());
        $this->assertEquals($resultValiable->shape(),$dSalt->shape());
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dOutputs),$K->ndarray($salt)
        ));
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dSalt),$K->ndarray($outputs)
        ));

        $copydOutputs = $K->copy($dOutputs);
        $dInputs = $outputsVariable->creator()->backward([$dOutputs]);
        // 2 batch
        $this->assertCount(2,$dInputs);
        //echo "dQuery: ".$mo->shapeToString($dInputs[0]->shape())."\n";
        //echo "dValue: ".$mo->shapeToString($dInputs[1]->shape())."\n";
        $this->assertEquals($full_query_shape,$dInputs[0]->shape());
        $this->assertEquals($full_value_shape,$dInputs[1]->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());

        //echo "dQuery: ".$mo->toString($dInputs[0],format:'%12.7e',indent:true)."\n";
        //echo "dValue: ".$mo->toString($dInputs[1],format:'%12.7e',indent:true)."\n";

        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dInputs[0]),
            $mo->array([
               [[  99.330315,   99.330315,   99.330315,   99.330315,   99.330315],
                [ 297.12323 ,  297.12323 ,  297.12323 ,  297.12323 ,  297.12323 ],
                [ 389.20917 ,  389.20917 ,  389.20917 ,  389.20917 ,  389.20917 ],
                [2775.5093  , 2775.5093  , 2775.5093  , 2775.5093  , 2775.5093  ],
                [3591.8362  , 3591.8362  , 3591.8362  , 3591.8362  , 3591.8362  ],
                [4408.163   , 4408.163   , 4408.163   , 4408.163   , 4408.163   ],],
              
               [[ 247.50195,   247.50195,   247.50195,   247.50195,   247.50195 ],
                [ 195.90976,   195.90976,   195.90976,   195.90976,   195.90976 ],
                [ 155.72949,   155.72949,   155.72949,   155.72949,   155.72949 ],
                [ 124.00012,   124.00012,   124.00012,   124.00012,   124.00012 ],
                [  98.75073,    98.75073,    98.75073,    98.75073,    98.75073 ],
                [4653.039  ,  4653.039  ,  4653.039  ,  4653.039  ,  4653.039   ],],
            ]),
            rtol:($srvLvl==Service::LV_BASIC)?1e+1:1e-4,
            //debug:true,
        ));
        $this->assertTrue($mo->la()->isclose(
            $K->ndarray($dInputs[1]),
            $mo->array([
              [[-4.9332588e+03, -4.9332588e+03, -4.9332588e+03, -4.9332588e+03, -4.9332588e+03],
               [-2.6076963e+03, -2.6076963e+03, -2.6076963e+03, -2.6076963e+03, -2.6076963e+03],
               [ 1.3684351e+02,  1.3684351e+02,  1.3684351e+02,  1.3684351e+02,  1.3684351e+02],
               [ 3.9983945e+03,  3.9983945e+03,  3.9983945e+03,  3.9983945e+03,  3.9983945e+03],
               [ 3.6419038e+03,  3.6419038e+03,  3.6419038e+03,  3.6419038e+03,  3.6419038e+03],
               [ 5.7752368e+03,  5.7752368e+03,  5.7752368e+03,  5.7752368e+03,  5.7752368e+03],
               [ 7.9085723e+03,  7.9085723e+03,  7.9085723e+03,  7.9085723e+03,  7.9085723e+03],],
             
              [[-2.9573274e+00, -2.9573274e+00, -2.9573274e+00, -2.9573274e+00, -2.9573274e+00],
               [-1.9088331e+01, -1.9088331e+01, -1.9088331e+01, -1.9088331e+01, -1.9088331e+01],
               [-1.0892528e+04, -1.0892528e+04, -1.0892528e+04, -1.0892528e+04, -1.0892528e+04],
               [-5.2124438e+03, -5.2124438e+03, -5.2124438e+03, -5.2124438e+03, -5.2124438e+03],
               [-1.1397036e+03, -1.1397036e+03, -1.1397036e+03, -1.1397036e+03, -1.1397036e+03],
               [ 4.5568398e+04,  4.5568398e+04,  4.5568398e+04,  4.5568398e+04,  4.5568398e+04],
               [ 1.4418279e+04,  1.4418279e+04,  1.4418279e+04,  1.4418279e+04,  1.4418279e+04],],
            ]),
            rtol:($srvLvl==Service::LV_BASIC)?1e+0:1e-4,
            //debug:true,
        ));

    }
}
