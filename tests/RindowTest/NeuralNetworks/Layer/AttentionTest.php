<?php
namespace RindowTest\NeuralNetworks\Layer\AttentionTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Layer\Attention;
use Rindow\NeuralNetworks\Gradient\Core\MaskedNDArray;
use InvalidArgumentException;
use WeakMap;

class AttentionTest extends TestCase
{
    public function newMatrixOperator()
    {
        return new MatrixOperator();
    }

    public function newNeuralNetworks($mo)
    {
        return new NeuralNetworks($mo);
    }

    public function testDefaultInitialize()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new Attention($K, input_shapes:[[3,2],[4,2]]);
        $inputs = [
            $g->Variable($K->zeros([1,3,2])),
            $g->Variable($K->zeros([1,4,2])),
        ];

        $layer->build($inputs);
        $params = $layer->getParams();
        $this->assertCount(0,$params);

        $grads = $layer->getGrads();
        $this->assertCount(0,$grads);

        $this->assertEquals([3,2],$layer->outputShape());
    }

    public function testInitializeWithReturnAttentionScores()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new Attention($K, input_shapes:[[3,2],[4,2]]);
        $inputs = [
            $g->Variable($K->zeros([1,3,2])),
            $g->Variable($K->zeros([1,4,2])),
        ];

        $shapes = $layer->build($inputs);
        $params = $layer->getParams();
        $this->assertCount(0,$params);

        $grads = $layer->getGrads();
        $this->assertCount(0,$grads);

        $this->assertEquals([3,2],$layer->outputShape());
    }

    public function testSetInputShape()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new Attention($K, input_shapes:[[3,2],[4,2]]);
        $inputs = [
            $g->Variable($K->zeros([1,3,2])),
            $g->Variable($K->zeros([1,4,2])),
        ];
        // [batch,3,2],[batch,4,2]
        $layer->build($inputs);
        // [batch,3,4]
        $this->assertEquals([3,2],$layer->outputShape());
    }

    public function testUnmatchSpecifiedInputShape()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new Attention($K, input_shapes:[[3,2],[4,2]]);
        $inputs = [
            $g->Variable($K->zeros([1,5,2])),
            $g->Variable($K->zeros([1,4,2])),
        ];
    
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Input shape is inconsistent: defined as ((3,2),(4,2)) but ((5,2),(4,2)) given in Attention');
        $layer->build($inputs);
    }

    public function testNormalForwardAndBackward()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new Attention($K);
        $inputs = [
            $g->Variable($K->zeros([2,2,3])),
            $g->Variable($K->zeros([2,4,3])),
        ];

        $layer->build($inputs);

        //
        // forward
        //
        //  batch size 2
        $query = $K->array([
            [[1,0,0],[0,1,0]],
            [[1,0,0],[0,1,0]],
        ]);
        $value = $K->array([
            [[1,0,0],[0,1,0],[0,0,1],[0,0,0]],
            [[1,0,0],[0,1,0],[0,0,1],[0,0,0]],
        ]);
        $inputs = [$query,$value];
        $copyInputs = [$K->copy($query),$K->copy($value)];
        [$outputsVariable,$scores] = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$inputs) {
                [$outputsVariable,$scores] = $layer->forward($inputs,
                                returnAttentionScores:true);
                return [$outputsVariable,$scores];
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        //
        $this->assertEquals([2,2,4],$scores->shape());
        $this->assertEquals([2,2,3],$outputs->shape());
        $this->assertEquals($copyInputs[0]->toArray(),$inputs[0]->toArray());
        $this->assertEquals($copyInputs[1]->toArray(),$inputs[1]->toArray());
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->softmax($mo->array([
                [1,0,0,0],[0,1,0,0],
                [1,0,0,0],[0,1,0,0],
            ])),
            $K->ndarray($scores->reshape([4,4]))));
        $this->assertTrue($mo->la()->isclose(
            $mo->array([
                [[0.475367,0.174878,0.174878],[0.174878,0.475367,0.174878]],
                [[0.475367,0.174878,0.174878],[0.174878,0.475367,0.174878]],
            ]),
            $K->ndarray($outputs)));
        //
        // backward
        //
        // 2 batch
        $dOutputs = $K->ones($outputs->shape());

        $copydOutputs = $K->copy(
            $dOutputs);
        $dInputs = $outputsVariable->creator()->backward([$dOutputs]);
        // 2 batch
        $this->assertCount(2,$dInputs);
        $this->assertEquals([2,2,3],$dInputs[0]->shape());
        $this->assertEquals([2,4,3],$dInputs[1]->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());
        $this->assertTrue($mo->la()->isclose(
            $mo->array([
                [[0.08313105, 0.0305822 , 0.0305822],
                 [0.0305822 , 0.08313105, 0.0305822]],
                [[0.08313105, 0.0305822 , 0.0305822],
                 [0.0305822 , 0.08313105, 0.0305822]],
            ]),
            $K->ndarray($dInputs[0])));
        $this->assertTrue($mo->la()->isclose(
            $mo->array([
                [[0.73337567, 0.68082684, 0.65024465],
                 [0.68082684, 0.73337567, 0.65024465],
                 [0.38033763, 0.38033763, 0.34975544],
                 [0.20545992, 0.20545992, 0.34975544]],
                [[0.73337567, 0.68082684, 0.65024465],
                 [0.68082684, 0.73337567, 0.65024465],
                 [0.38033763, 0.38033763, 0.34975544],
                 [0.20545992, 0.20545992, 0.34975544]],
            ]),
            $K->ndarray($dInputs[1])));
    }

    public function testMaskBoth()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new Attention($K);
        $inputs = [
            $g->Variable($K->zeros([2,3,4])),
            $g->Variable($K->zeros([2,5,4])),
        ];

        $layer->build($inputs);

        //
        // forward
        //
        //  batch size 2
        $query = $K->ones([2,3,4]);
        $value = $K->ones([2,5,4]);
        $queryMask = $K->array([ // (2,3)
            [true,true, false],
            [true,false,false],
        ],dtype:NDArray::bool);
        $valueMask = $K->array([ // (2,5)
            [true,true,false,false,false],
            [true,true,true, true, false],
        ],dtype:NDArray::bool);
        $inputs = [$query,$value];
        $copyInputs = [$K->copy($query),$K->copy($value)];
        [$outputsVariable,$scores] = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$inputs,$queryMask,$valueMask) {
                [$outputsVariable,$scores] = $layer->forward($inputs, mask:[$queryMask,$valueMask],
                                returnAttentionScores:true);
                return [$outputsVariable,$scores];
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        //
        $this->assertEquals([2,3,5],$scores->shape());
        $this->assertEquals([2,3,4],$outputs->shape());
        $this->assertEquals($copyInputs[0]->toArray(),$inputs[0]->toArray());
        $this->assertEquals($copyInputs[1]->toArray(),$inputs[1]->toArray());
        $this->assertTrue($mo->la()->isclose(
            $mo->array(
                [[[0.5, 0.5, 0.0, 0.0, 0.0],
                  [0.5, 0.5, 0.0, 0.0, 0.0],
                  [0.5, 0.5, 0.0, 0.0, 0.0]],
                  [[0.25, 0.25, 0.25, 0.25, 0.0  ],
                   [0.25, 0.25, 0.25, 0.25, 0.0  ],
                   [0.25, 0.25, 0.25, 0.25, 0.0  ]]]
            ),
            $scores = $K->ndarray($scores)
        ));
        $this->assertTrue($mo->la()->isclose(
            $mo->array(
                [[[1.0, 1.0, 1.0, 1.0],
                  [1.0, 1.0, 1.0, 1.0],
                  [0.0, 0.0, 0.0, 0.0]],
                 [[1.0, 1.0, 1.0, 1.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0]]]
            ),
            $outputs = $K->ndarray($outputs)
        ));
        //
        // backward
        //
        // 2 batch
        $dOutputs = $K->ones($outputs->shape());

        $copydOutputs = $K->copy(
            $dOutputs);
        $dInputs = $outputsVariable->creator()->backward([$dOutputs]);
        // 2 batch
        $this->assertCount(2,$dInputs);
        $this->assertEquals([2,3,4],$dInputs[0]->shape());
        $this->assertEquals([2,5,4],$dInputs[1]->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());
        $this->assertTrue($mo->la()->isclose(
            $mo->array(
                [[[0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0]],
                 [[0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0]]]
            ),
            $K->ndarray($dInputs[0])));
        $this->assertTrue($mo->la()->isclose(
            $mo->array(
                [[[1.0,   1.0,   1.0,   1.0 ],
                  [1.0,   1.0,   1.0,   1.0 ],
                  [0.0,   0.0,   0.0,   0.0 ],
                  [0.0,   0.0,   0.0,   0.0 ],
                  [0.0,   0.0,   0.0,   0.0 ]],
                 [[0.25,  0.25,  0.25,  0.25],
                  [0.25,  0.25,  0.25,  0.25],
                  [0.25,  0.25,  0.25,  0.25],
                  [0.25,  0.25,  0.25,  0.25],
                  [0.0,   0.0,   0.0,   0.0 ]]]
            ),
            $K->ndarray($dInputs[1])));
    }

    public function testMaskBothMaskedValue()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new Attention($K);
        $inputs = [
            $g->Variable($K->zeros([2,3,4])),
            $g->Variable($K->zeros([2,5,4])),
        ];

        $layer->build($inputs);

        //
        // forward
        //
        //  batch size 2
        $query = $K->ones([2,3,4]);
        $value = $K->ones([2,5,4]);
        $queryMask = $K->array([ // (2,3)
            [true,true, false],
            [true,false,false],
        ],dtype:NDArray::bool);
        $valueMask = $K->array([ // (2,5)
            [true,true,false,false,false],
            [true,true,true, true, false],
        ],dtype:NDArray::bool);
        $inputs = [
            new MaskedNDArray($query,$queryMask),
            new MaskedNDArray($value,$valueMask),
        ];
        $copyInputs = [$K->copy($query),$K->copy($value)];
        [$outputsVariable,$scores] = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$inputs) {
                [$outputsVariable,$scores] = $layer->forward($inputs,
                                returnAttentionScores:true);
                return [$outputsVariable,$scores];
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        //
        $this->assertEquals([2,3,5],$scores->shape());
        $this->assertEquals([2,3,4],$outputs->shape());
        $this->assertEquals($copyInputs[0]->toArray(),$inputs[0]->toArray());
        $this->assertEquals($copyInputs[1]->toArray(),$inputs[1]->toArray());
        $this->assertTrue($mo->la()->isclose(
            $mo->array(
                [[[0.5, 0.5, 0.0, 0.0, 0.0],
                  [0.5, 0.5, 0.0, 0.0, 0.0],
                  [0.5, 0.5, 0.0, 0.0, 0.0]],
                  [[0.25, 0.25, 0.25, 0.25, 0.0  ],
                   [0.25, 0.25, 0.25, 0.25, 0.0  ],
                   [0.25, 0.25, 0.25, 0.25, 0.0  ]]]
            ),
            $scores = $K->ndarray($scores)
        ));
        $this->assertTrue($mo->la()->isclose(
            $mo->array(
                [[[1.0, 1.0, 1.0, 1.0],
                  [1.0, 1.0, 1.0, 1.0],
                  [0.0, 0.0, 0.0, 0.0]],
                 [[1.0, 1.0, 1.0, 1.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0]]]
            ),
            $outputs = $K->ndarray($outputs)
        ));
        //
        // backward
        //
        // 2 batch
        $dOutputs = $K->ones($outputs->shape());

        $copydOutputs = $K->copy(
            $dOutputs);
        $dInputs = $outputsVariable->creator()->backward([$dOutputs]);
        // 2 batch
        $this->assertCount(2,$dInputs);
        $this->assertEquals([2,3,4],$dInputs[0]->shape());
        $this->assertEquals([2,5,4],$dInputs[1]->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());
        $this->assertTrue($mo->la()->isclose(
            $mo->array(
                [[[0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0]],
                 [[0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0]]]
            ),
            $K->ndarray($dInputs[0])));
        $this->assertTrue($mo->la()->isclose(
            $mo->array(
                [[[1.0,   1.0,   1.0,   1.0 ],
                  [1.0,   1.0,   1.0,   1.0 ],
                  [0.0,   0.0,   0.0,   0.0 ],
                  [0.0,   0.0,   0.0,   0.0 ],
                  [0.0,   0.0,   0.0,   0.0 ]],
                 [[0.25,  0.25,  0.25,  0.25],
                  [0.25,  0.25,  0.25,  0.25],
                  [0.25,  0.25,  0.25,  0.25],
                  [0.25,  0.25,  0.25,  0.25],
                  [0.0,   0.0,   0.0,   0.0 ]]]
            ),
            $K->ndarray($dInputs[1])));
    }

/*
    public function testMaskFloatBoth()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new Attention($K);
        $inputs = [
            $g->Variable($K->zeros([2,3,4])),
            $g->Variable($K->zeros([2,5,4])),
        ];

        $layer->build($inputs);

        //
        // forward
        //
        //  batch size 2
        $query = $K->ones([2,3,4]);
        $value = $K->ones([2,5,4]);
        $queryMask = $K->array([ // (2,3)
            [true, true, false],
            [true, true, false],
        ]);
        $valueMask = $K->array([ // (2,5)
            [true, true, false, false, false],
            [true, true, true,  true,  false],
        ]);
        $inputs = [$query,$value];
        $copyInputs = [$K->copy($query),$K->copy($value)];
        [$outputsVariable,$scores] = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$inputs,$queryMask,$valueMask) {
                [$outputsVariable,$scores] = $layer->forward($inputs, mask:[$queryMask,$valueMask],
                                returnAttentionScores:true);
                return [$outputsVariable,$scores];
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        //
        $this->assertEquals([2,3,5],$scores->shape());
        $this->assertEquals([2,3,4],$outputs->shape());
        $this->assertEquals($copyInputs[0]->toArray(),$inputs[0]->toArray());
        $this->assertEquals($copyInputs[1]->toArray(),$inputs[1]->toArray());
        $this->assertTrue($mo->la()->isclose(
            $mo->array(
                [[[0.5, 0.5, 0.0, 0.0, 0.0],
                  [0.5, 0.5, 0.0, 0.0, 0.0],
                  [0.5, 0.5, 0.0, 0.0, 0.0]],
                  [[0.25, 0.25, 0.25, 0.25, 0.0  ],
                   [0.25, 0.25, 0.25, 0.25, 0.0  ],
                   [0.25, 0.25, 0.25, 0.25, 0.0  ]]]
            ),
            $scores = $K->ndarray($scores)
        ));
        $this->assertTrue($mo->la()->isclose(
            $mo->array(
                [[[1.0, 1.0, 1.0, 1.0],
                  [1.0, 1.0, 1.0, 1.0],
                  [0.0, 0.0, 0.0, 0.0]],
                 [[1.0, 1.0, 1.0, 1.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0]]]
            ),
            $outputs = $K->ndarray($outputs)
        ));
        //
        // backward
        //
        // 2 batch
        $dOutputs = $K->ones($outputs->shape());

        $copydOutputs = $K->copy(
            $dOutputs);
        $dInputs = $outputsVariable->creator()->backward([$dOutputs]);
        // 2 batch
        $this->assertCount(2,$dInputs);
        $this->assertEquals([2,3,4],$dInputs[0]->shape());
        $this->assertEquals([2,5,4],$dInputs[1]->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());
        $this->assertTrue($mo->la()->isclose(
            $mo->array(
                [[[0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0]],
                 [[0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0]]]
            ),
            $K->ndarray($dInputs[0])));
        $this->assertTrue($mo->la()->isclose(
            $mo->array(
                [[[1.0,   1.0,   1.0,   1.0 ],
                  [1.0,   1.0,   1.0,   1.0 ],
                  [0.0,   0.0,   0.0,   0.0 ],
                  [0.0,   0.0,   0.0,   0.0 ],
                  [0.0,   0.0,   0.0,   0.0 ]],
                 [[0.25,  0.25,  0.25,  0.25],
                  [0.25,  0.25,  0.25,  0.25],
                  [0.25,  0.25,  0.25,  0.25],
                  [0.25,  0.25,  0.25,  0.25],
                  [0.0,   0.0,   0.0,   0.0 ]]]
            ),
            $K->ndarray($dInputs[1])));
    }

    public function testMaskDoNotExpandMask()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new Attention($K,do_not_expand_mask:true);

        $query = $K->ones([2,2,3,2]);
        $value = $K->ones([2,2,5,2]);

        $inputs = [
            $g->Variable($query),
            $g->Variable($value),
        ];
        $layer->build($inputs);

        //
        // forward
        //
        //  batch size 2
        $queryMask = $K->array([ // (2,1,3,1)
            [[[true],[true], [false]]],
            [[[true],[false],[false]]],
        ],dtype:NDArray::bool);
        $valueMask = $K->array([ // (2,1,1,5)
            [[[true,true,false,false,false]]],
            [[[true,true,true, true, false]]],
        ],dtype:NDArray::bool);
        $inputs = [$query,$value];
        $copyInputs = [$K->copy($query),$K->copy($value)];
        [$outputsVariable,$scores] = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$inputs,$queryMask,$valueMask) {
                [$outputsVariable,$scores] = $layer->forward($inputs, mask:[$queryMask,$valueMask],
                                returnAttentionScores:true);
                return [$outputsVariable,$scores];
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        //
        $this->assertEquals([2,2,3,5],$scores->shape());
        $this->assertEquals([2,2,3,2],$outputs->shape());
        $this->assertEquals($copyInputs[0]->toArray(),$inputs[0]->toArray());
        $this->assertEquals($copyInputs[1]->toArray(),$inputs[1]->toArray());
        //
        // backward
        //
        // 2 batch
        $dOutputs = $K->ones($outputs->shape());

        $copydOutputs = $K->copy(
            $dOutputs);
        $dInputs = $outputsVariable->creator()->backward([$dOutputs]);
        // 2 batch
        $this->assertCount(2,$dInputs);
        $this->assertEquals([2,2,3,2],$dInputs[0]->shape());
        $this->assertEquals([2,2,5,2],$dInputs[1]->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());
    }
*/

    public function testMaskOneSide()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new Attention($K);
        $inputs = [
            $g->Variable($K->zeros([2,2,3])),
            $g->Variable($K->zeros([2,4,3])),
        ];

        $layer->build($inputs);

        $query = $K->array([
            [[1,0,0],[0,1,0]],
            [[1,0,0],[0,1,0]],
        ]);
        $value = $K->array([
            [[1,0,0],[0,1,0],[0,0,1],[0,0,0]],
            [[1,0,0],[0,1,0],[0,0,1],[0,0,0]],
        ]);
        $queryMask = $K->array([
            [true,false],
            [false,true],
        ],dtype:NDArray::bool);
        $valueMask = $K->array([
            [false,false,true,true],
            [false,true,true,false],
        ],dtype:NDArray::bool);
        $inputs = [$query,$value];
        $copyInputs = [$K->copy($query),$K->copy($value)];

        //
        //  queryMask
        //
        // forward
        //
        [$outputsVariable,$scores] = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$inputs,$queryMask,$valueMask) {
                [$outputsVariable,$scores] = $layer->forward($inputs, mask:[$queryMask,null],
                                returnAttentionScores:true);
                return [$outputsVariable,$scores];
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        //
        $this->assertEquals([2,2,4],$scores->shape());
        $this->assertEquals([2,2,3],$outputs->shape());
        $this->assertEquals($copyInputs[0]->toArray(),$inputs[0]->toArray());
        $this->assertEquals($copyInputs[1]->toArray(),$inputs[1]->toArray());
        //
        // backward
        //
        $dOutputs = $K->ones($outputs->shape());

        $copydOutputs = $K->copy(
            $dOutputs);
        $dInputs = $outputsVariable->creator()->backward([$dOutputs]);
        // 2 batch
        $this->assertCount(2,$dInputs);
        $this->assertEquals([2,2,3],$dInputs[0]->shape());
        $this->assertEquals([2,4,3],$dInputs[1]->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());


        //
        //  valueMask
        //
        // forward
        //
        [$outputsVariable,$scores] = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$inputs,$queryMask,$valueMask) {
                [$outputsVariable,$scores] = $layer->forward($inputs, mask:[null,$valueMask],
                                returnAttentionScores:true);
                return [$outputsVariable,$scores];
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        //
        $this->assertEquals([2,2,4],$scores->shape());
        $this->assertEquals([2,2,3],$outputs->shape());
        $this->assertEquals($copyInputs[0]->toArray(),$inputs[0]->toArray());
        $this->assertEquals($copyInputs[1]->toArray(),$inputs[1]->toArray());
        //
        // backward
        //
        $dOutputs = $K->ones($outputs->shape());

        $copydOutputs = $K->copy(
            $dOutputs);
        $dInputs = $outputsVariable->creator()->backward([$dOutputs]);
        // 2 batch
        $this->assertCount(2,$dInputs);
        $this->assertEquals([2,2,3],$dInputs[0]->shape());
        $this->assertEquals([2,4,3],$dInputs[1]->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());
    }

    public function testUseScale()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new Attention($K,use_scale:true);
        $inputs = [
            $g->Variable($K->zeros([2,2,3])),
            $g->Variable($K->zeros([2,4,3])),
        ];
        $layer->build($inputs);

        //
        // forward
        //
        $query = $K->array(
            [[[1,0,0],[0,1,0]],
             [[1,0,0],[0,1,0]]]
        );
        $value = $K->array(
            [[[1,0,0],[0,1,0],[0,0,1],[0,0,0]],
             [[1,0,0],[0,1,0],[0,0,1],[0,0,0]]]
        );
        $inputs = [$query,$value];
        $copyInputs = [$K->copy($query),$K->copy($value)];
        [$outputsVariable,$scoresVariable] = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$inputs) {
                [$outputsVariable,$scoresVariable] = $layer->forward($inputs,
                                returnAttentionScores:true);
                return [$outputsVariable,$scoresVariable];
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        $scores = $K->ndarray($scoresVariable);

        //
        $this->assertEquals([2,2,4],$scores->shape());
        $this->assertEquals([2,2,3],$outputs->shape());
        $this->assertEquals($copyInputs[0]->toArray(),$inputs[0]->toArray());
        $this->assertEquals($copyInputs[1]->toArray(),$inputs[1]->toArray());
        $this->assertTrue($mo->la()->isclose(
            $mo->array(
                [[[0.47536692, 0.17487772, 0.17487772, 0.17487772],
                  [0.17487772, 0.47536692, 0.17487772, 0.17487772]],
                 [[0.47536692, 0.17487772, 0.17487772, 0.17487772],
                  [0.17487772, 0.47536692, 0.17487772, 0.17487772]]]
            ),
            $K->ndarray($scores)
        ));
        $this->assertTrue($mo->la()->isclose(
            $mo->array(
                [[[0.47536692, 0.17487772, 0.17487772],
                  [0.17487772, 0.47536692, 0.17487772]],
                 [[0.47536692, 0.17487772, 0.17487772],
                  [0.17487772, 0.47536692, 0.17487772]]]
            ),
            $K->ndarray($outputs)
        ));

        //
        // backward
        //
        $dOutputs = [
            $K->ones($outputs->shape()),
            $K->ones($scores->shape()),
        ];

        $variables = $layer->trainableVariables();
        $grads = new WeakMap();
        $copydOutputs = [];
        $copydOutputs[] = $K->copy($dOutputs[0]);
        $copydOutputs[] = $K->copy($dOutputs[1]);
        $dInputs = $outputsVariable->creator()->backward($dOutputs,$grads,$variables);
        // 2 batch
        $this->assertCount(2,$dInputs);
        $this->assertEquals([2,2,3],$dInputs[0]->shape());
        $this->assertEquals([2,4,3],$dInputs[1]->shape());
        $this->assertEquals($copydOutputs[0]->toArray(),$dOutputs[0]->toArray());
        $this->assertEquals($copydOutputs[1]->toArray(),$dOutputs[1]->toArray());

        $this->assertCount(1,$variables);
        $this->assertCount(1,$grads);
        $this->assertEquals(1.0, $K->scalar($variables[0]->value()));
        $this->assertLessThan(1e-6,(0.33252418-$K->scalar($grads[$variables[0]])));
    }

    public function testMultiHead()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new Attention($K,use_scale:true);
        $inputs = [
            $g->Variable($K->zeros([2,4,2,2,3])),
            $g->Variable($K->zeros([2,4,2,4,3])),
        ];
        $layer->build($inputs);

        //
        // forward
        //
        $query = $K->ones(
            $inputs[0]->shape()
        );
        $value = $K->ones(
            $inputs[1]->shape()
        );
        $inputs = [$query,$value];
        $copyInputs = [$K->copy($query),$K->copy($value)];
        [$outputsVariable,$scoresVariable] = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$inputs) {
                [$outputsVariable,$scoresVariable] = $layer->forward($inputs,
                                returnAttentionScores:true);
                return [$outputsVariable,$scoresVariable];
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        $scores = $K->ndarray($scoresVariable);

        //
        $this->assertEquals([2,4,2,2,4],$scores->shape());
        $this->assertEquals([2,4,2,2,3],$outputs->shape());
        //
        // backward
        //
        $dOutputs = [
            $K->ones($outputs->shape()),
            $K->ones($scores->shape()),
        ];

        $variables = $layer->trainableVariables();
        $grads = new WeakMap();
        $copydOutputs = [];
        $copydOutputs[] = $K->copy($dOutputs[0]);
        $copydOutputs[] = $K->copy($dOutputs[1]);
        $dInputs = $outputsVariable->creator()->backward($dOutputs,$grads,$variables);
        // 2 batch
        $this->assertCount(2,$dInputs);
        $this->assertEquals([2,4,2,2,3],$dInputs[0]->shape());
        $this->assertEquals([2,4,2,4,3],$dInputs[1]->shape());

        $this->assertCount(1,$variables);
        $this->assertCount(1,$grads);
    }

}
