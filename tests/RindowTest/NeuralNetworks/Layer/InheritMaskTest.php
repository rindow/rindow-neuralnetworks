<?php
namespace RindowTest\NeuralNetworks\Layer\InheritMaskTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Layer\InheritMask;
use Rindow\NeuralNetworks\Gradient\MaskedNDArray;
use Rindow\NeuralNetworks\Gradient\Core\MaskedNDArray as MaskedNDArrayImpl;
use InvalidArgumentException;

class InheritMaskTest extends TestCase
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

    public function testDefaultInitialize()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new InheritMask($K,
                input_shapes:[[4,3],[4,2]],
        );
        $inputs = [
            $g->Variable($K->zeros([1,4,3])),
            $g->Variable($this->maskedValue($K->zeros([1,4,2]),$K->ones([1,4]))),
        ];

        $layer->build($inputs);
        $params = $layer->getParams();
        $this->assertCount(0,$params);

        $grads = $layer->getGrads();
        $this->assertCount(0,$grads);

        $this->assertEquals([4,3],$layer->outputShape());
    }

    public function testUnmatchSpecifiedInputShape()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new InheritMask($K,input_shapes:[[4,3],[4,2]]);
        $inputs = [
            $g->Variable($K->zeros([1,4,5])),
            $g->Variable($this->maskedValue($K->zeros([1,4,2]),$K->ones([1,4]))),
        ];
    
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Input shape is inconsistent: defined as ((4,3),(4,2)) but ((4,5),(4,2)) given in InheritMask');
        $layer->build($inputs);
    }

    public function testSetInputShape()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new InheritMask($K);
        // [batch,2,4],[batch,3,4]
        $inputs = [
            $g->Variable($K->zeros([1,2,4])),
            $g->Variable($this->maskedValue($K->zeros([1,3,4]),$K->ones([1,3]))),
        ];
        $layer->build($inputs);
        // [batch,5,4]
        $this->assertEquals([2,4],$layer->outputShape());
    }

    public function testNormalForwardAndBackward()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();

        $layer = new InheritMask($K,
                #axis:-1,
        );

        //  batch size 2
        $i1 = $K->array($mo->arange(2*2*2,null,null,NDArray::float32)->reshape([2,2,2]));
        $i2 = $this->maskedValue(
            $K->array($mo->arange(2*2*3,100,null,NDArray::float32)->reshape([2,2,3])),
            $K->array([[true,true],[true,false]]),
        );
        $inputs = [$i1,$i2];

        $layer->build([$g->Variable($i1),$g->Variable($i2)]);

        //
        // forward
        //
        $copyInputs = [$K->copy($i1),$K->copy($i2)];
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$inputs) {
                $outputsVariable = $layer->forward($inputs);
                return $outputsVariable;
            }
        );
        $this->assertInstanceof(MaskedNDArray::class,$outputsVariable->value());
        $this->assertEquals([2,2],$outputsVariable->value()->mask()->shape());
        $this->assertEquals([[true,true],[true,false]],$outputsVariable->value()->mask()->toArray());
        $outputs = $K->ndarray($outputsVariable);
        //
        $this->assertEquals([2,2,2],$outputs->shape());
        $this->assertEquals($copyInputs[0]->toArray(),$inputs[0]->toArray());
        $this->assertEquals($copyInputs[1]->toArray(),$inputs[1]->toArray());
        $this->assertEquals($copyInputs[0]->toArray(),$outputs->toArray());
        //
        // backward
        //
        // 2 batch
        $dOutputs = $K->copy($outputsVariable->value());

        $copydOutputs = $K->copy(
            $dOutputs);
        $dInputs = $outputsVariable->creator()->backward([$dOutputs]);
        // 2 batch
        $this->assertCount(2,$dInputs);
        $this->assertEquals([2,2,2],$dInputs[0]->shape());
        $this->assertEquals([2,2,3],$dInputs[1]->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());
        $this->assertEquals($dOutputs->toArray(),$dInputs[0]->toArray());
        $this->assertEquals($K->zeros([2,2,3])->toArray(),$dInputs[1]->toArray());
    }

    public function testReplaceMask()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();

        $layer = new InheritMask($K,
                #axis:-1,
        );

        //  batch size 2
        $i1 = $this->maskedValue(
            $K->array($mo->arange(2*2*2,null,null,NDArray::float32)->reshape([2,2,2])),
            $K->array([[true,false],[false,false],[false,false]]),
        );
        $i2 = $this->maskedValue(
            $K->array($mo->arange(2*2*3,100,null,NDArray::float32)->reshape([2,2,3])),
            $K->array([[true,true],[true,false]]),
        );
        $inputs = [$i1,$i2];

        $layer->build([$g->Variable($i1),$g->Variable($i2)]);

        //
        // forward
        //
        $copyInputs = [$K->copy($i1),$K->copy($i2)];
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$inputs) {
                $outputsVariable = $layer->forward($inputs);
                return $outputsVariable;
            }
        );
        $this->assertInstanceof(MaskedNDArray::class,$outputsVariable->value());
        $this->assertEquals([2,2],$outputsVariable->value()->mask()->shape());
        $this->assertEquals([[true,true],[true,false]],$outputsVariable->value()->mask()->toArray());
        $outputs = $K->ndarray($outputsVariable);
        //
        $this->assertEquals([2,2,2],$outputs->shape());
        $this->assertEquals($copyInputs[0]->toArray(),$inputs[0]->toArray());
        $this->assertEquals($copyInputs[1]->toArray(),$inputs[1]->toArray());
        $this->assertEquals($copyInputs[0]->toArray(),$outputs->toArray());
        //
        // backward
        //
        // 2 batch
        $dOutputs = $K->copy($outputsVariable->value());

        $copydOutputs = $K->copy(
            $dOutputs);
        $dInputs = $outputsVariable->creator()->backward([$dOutputs]);
        // 2 batch
        $this->assertCount(2,$dInputs);
        $this->assertEquals([2,2,2],$dInputs[0]->shape());
        $this->assertEquals([2,2,3],$dInputs[1]->shape());
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());
        $this->assertEquals($dOutputs->toArray(),$dInputs[0]->toArray());
        $this->assertEquals($K->zeros([2,2,3])->toArray(),$dInputs[1]->toArray());
    }
}
