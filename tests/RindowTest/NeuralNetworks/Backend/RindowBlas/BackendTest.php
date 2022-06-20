<?php
namespace RindowTest\NeuralNetworks\Backend\RindowBlas\BackendTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use SplFixedArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use InvalidArgumentException;

class Test extends TestCase
{
    public function newMatrixOperator()
    {
        return new MatrixOperator();
    }

    public function newBackend($mo)
    {
        return new Backend($mo);
    }

    public function getPlotConfig()
    {
        return [
            'renderer.skipCleaning' => true,
            'renderer.skipRunViewer' => getenv('TRAVIS_PHP_VERSION') ? true : false,
        ];
    }

    public function testGetInitializer()
    {
        $mo = $this->newMatrixOperator();
        $backend = $this->newBackend($mo);
        $backendClassName = get_class($backend);

        $initializer = $backend->getInitializer('he_normal');
        $this->assertInstanceof(
            $backendClassName,
            $initializer[0]
        );
        $this->assertEquals('he_normal',$initializer[1]);
        $this->assertTrue(is_callable($initializer));
        $kernel = $initializer([2,3]);
        $this->assertInstanceof('Interop\Polite\Math\Matrix\NDArray',$kernel);
        $this->assertEquals([2,3],$kernel->shape());
        $this->assertEquals(NDArray::float32,$kernel->dtype());


        $initializer = $backend->getInitializer('glorot_normal');
        $this->assertInstanceof(
            $backendClassName,
            $initializer[0]
        );
        $this->assertEquals('glorot_normal',$initializer[1]);
        $this->assertTrue(is_callable($initializer));
        $kernel = $initializer([2,3]);
        $this->assertInstanceof('Interop\Polite\Math\Matrix\NDArray',$kernel);
        $this->assertEquals([2,3],$kernel->shape());
        $this->assertEquals(NDArray::float32,$kernel->dtype());


        $initializer = $backend->getInitializer('zeros');
        $this->assertInstanceof(
            $backendClassName,
            $initializer[0]
        );
        $this->assertEquals('zeros',$initializer[1]);
        $this->assertTrue(is_callable($initializer));
        $kernel = $initializer([2]);
        $this->assertInstanceof('Interop\Polite\Math\Matrix\NDArray',$kernel);
        $this->assertEquals([2],$kernel->shape());
        $this->assertEquals(NDArray::float32,$kernel->dtype());

        $initializer = $backend->getInitializer('ones');
        $this->assertInstanceof(
            $backendClassName,
            $initializer[0]
        );
        $this->assertEquals('ones',$initializer[1]);
        $this->assertTrue(is_callable($initializer));
        $kernel = $initializer([2]);
        $this->assertInstanceof('Interop\Polite\Math\Matrix\NDArray',$kernel);
        $this->assertEquals([2],$kernel->shape());
        $this->assertEquals(NDArray::float32,$kernel->dtype());
    }

    public function testUnsupportedInitializer()
    {
        $mo = $this->newMatrixOperator();
        $backend = $this->newBackend($mo);

        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Unsupported initializer: boo');
        $initializer = $backend->getInitializer('boo');
    }


    public function testZeros()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $this->assertEquals([[0,0]],$K->zeros([1,2])->toArray());
    }

    public function testOnes()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $this->assertEquals([[1,1]],$K->ones([1,2])->toArray());
    }

    public function testZerosLike()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $x = $mo->ones([1,2]);
        $this->assertEquals([[0,0]],$K->zerosLike($x)->toArray());
    }

    public function testOnesLike()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $x = $mo->zeros([1,2]);
        $this->assertEquals([[1,1]],$K->onesLike($x)->toArray());
    }

    public function testCast()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);

        $x = $K->ones([1,2],$dtype=NDArray::int32);
        $this->assertEquals(NDArray::int32,$x->dtype());

        $y = $K->cast($x,NDArray::float32); $K->finish();
        $this->assertEquals(NDArray::float32,$y->dtype());
        if(is_scalar($y[0][0])) $this->assertTrue(is_float($y[0][0]));

        $y = $K->cast($x,NDArray::bool); $K->finish();
        $this->assertEquals(NDArray::bool,$y->dtype());
        if(is_scalar($y[0][0])) $this->assertTrue(is_bool($y[0][0]));
        if(is_scalar($y[0][0])) $this->assertEquals(true,$y[0][0]);

        $x = $K->ones([1,2],$dtype=NDArray::float32);
        $this->assertEquals(NDArray::float32,$x->dtype());

        $y = $K->cast($x,NDArray::int32); $K->finish();
        $this->assertEquals(NDArray::int32,$y->dtype());
        if(is_scalar($y[0][0])) $this->assertTrue(is_int($y[0][0]));
        $y = $K->cast($x,NDArray::bool); $K->finish();
        $this->assertEquals(NDArray::bool,$y->dtype());
        if(is_scalar($y[0][0])) $this->assertTrue($y[0][0]);
        if(is_scalar($y[0][0])) $this->assertEquals(true,$y[0][0]);

        $x = $K->array([[true,false]],$dtype=NDArray::bool);
        $this->assertEquals(NDArray::bool,$x->dtype());
        $y = $K->cast($x,NDArray::int32); $K->finish();
        $this->assertEquals(NDArray::int32,$y->dtype());
        if(is_scalar($y[0][0])) $this->assertTrue(is_int($y[0][0]));
        if(is_scalar($y[0][0])) $this->assertEquals(1,$y[0][0]);
        if(is_scalar($y[0][0])) $this->assertEquals(0,$y[0][1]);
        $y = $K->cast($x,NDArray::float32); $K->finish();
        $this->assertEquals(NDArray::float32,$y->dtype());
        if(is_scalar($y[0][0])) $this->assertTrue(is_float($y[0][0]));
        if(is_scalar($y[0][0])) $this->assertEquals(1.0,$y[0][0]);
        if(is_scalar($y[0][0])) $this->assertEquals(0.0,$y[0][1]);
    }

    public function testAdd()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $x = $K->array([2,3]);
        $y = $K->array([10,20]);
        $z = $K->add($x,$y); $K->finish();
        $this->assertEquals([12,23],$z->toArray());
    }

    public function testSub()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $x = $K->array([10,20]);
        $y = $K->array([2,3]);
        $z = $K->sub($x,$y); $K->finish();
        $this->assertEquals([8,17],$z->toArray());
    }

    public function testMul()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $x = $K->array([10,20]);
        $y = $K->array([2,3]);
        $z = $K->mul($x,$y); $K->finish();
        $this->assertEquals([20,60],$z->toArray());
    }

    public function testDiv()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $x = $K->array([10,21]);
        $y = $K->array([2,3]);
        $z = $K->div($x,$y); $K->finish();
        $this->assertTrue($K->equalTest(
            $K->array([5,7]),$z));
    }

    public function testUpdate_add()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $x = $K->array([10,20]);
        $y = $K->array([2,3]);
        $z = $K->update_add($x,$y); $K->finish();
        $this->assertEquals([12,23],$z->toArray());
        $this->assertEquals([12,23],$x->toArray());
    }

    public function testUpdate_sub()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $x = $K->array([10,20]);
        $y = $K->array([2,3]);
        $z = $K->update_sub($x,$y); $K->finish();
        $this->assertEquals([8,17],$z->toArray());
        $this->assertEquals([8,17],$x->toArray());
    }

    public function testScale()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $x = $K->array([2,3]);
        $z = $K->scale(2,$x); $K->finish();
        $this->assertEquals([4,6],$z->toArray());
    }

    public function testPow()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $x = $K->array([2,3]);
        $z = $K->scale(6,$K->pow($x,-1)); $K->finish();
        $this->assertEquals([3,2],$z->toArray());
    }

    public function testSqrt()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $x = $K->array([4,9]);
        $z = $K->sqrt($x); $K->finish();
        $this->assertEquals([2,3],$z->toArray());
    }

    public function testGreater()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $x = $K->array([4,9]);
        $z = $K->greater($x,5); $K->finish();
        $this->assertEquals([0,1],$z->toArray());
    }

    public function testEqual()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $x = $K->array([4,9],NDArray::float32);
        $y = $K->array([9,9],NDArray::float32);
        $z = $K->equal($x,$y); $K->finish();
        $this->assertEquals([0,1],$z->toArray());
        $this->assertEquals(NDArray::float32,$z->dtype());

        $x = $K->array([4,9],NDArray::int32);
        $y = $K->array([9,9],NDArray::int32);
        $z = $K->equal($x,$y); $K->finish();
        $this->assertEquals([0,1],$z->toArray());
        $this->assertEquals(NDArray::int32,$z->dtype());
    }

    public function testSum()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);

        $x = $K->array([[true,false],[true,true]],NDArray::bool);
        $z = $K->sum($x); $K->finish(); if(!is_scalar($z)) $z = $z->toArray();
        $this->assertEquals(3,$z);
        $x = $K->array([[2,3],[1,2]],NDArray::int8);
        $z = $K->sum($x); $K->finish(); if(!is_scalar($z)) $z = $z->toArray();
        $this->assertEquals(8,$z);
        $x = $K->array([[2,3],[1,2]],NDArray::uint8);
        $z = $K->sum($x); $K->finish(); if(!is_scalar($z)) $z = $z->toArray();
        $this->assertEquals(8,$z);
        $x = $K->array([[2,3],[1,2]],NDArray::int16);
        $z = $K->sum($x); $K->finish(); if(!is_scalar($z)) $z = $z->toArray();
        $this->assertEquals(8,$z);
        $x = $K->array([[2,3],[1,2]],NDArray::uint16);
        $z = $K->sum($x); $K->finish(); if(!is_scalar($z)) $z = $z->toArray();
        $this->assertEquals(8,$z);
        $x = $K->array([[2,3],[1,2]],NDArray::int32);
        $z = $K->sum($x); $K->finish(); if(!is_scalar($z)) $z = $z->toArray();
        $this->assertEquals(8,$z);
        $x = $K->array([[2,3],[1,2]],NDArray::uint32);
        $z = $K->sum($x); $K->finish(); if(!is_scalar($z)) $z = $z->toArray();
        $this->assertEquals(8,$z);
        $x = $K->array([[2,3],[1,2]],NDArray::int64);
        $z = $K->sum($x); $K->finish(); if(!is_scalar($z)) $z = $z->toArray();
        $this->assertEquals(8,$z);
        $x = $K->array([[2,3],[1,2]],NDArray::uint64);
        $z = $K->sum($x); $K->finish(); if(!is_scalar($z)) $z = $z->toArray();
        $this->assertEquals(8,$z);
        $x = $K->array([[2,3],[1,2]],NDArray::float32);
        $z = $K->sum($x); $K->finish(); if(!is_scalar($z)) $z = $z->toArray();
        $this->assertEquals(8,$z);
        if($K->fp64()) {
            $x = $K->array([[2,3],[1,2]],NDArray::float64);
            $z = $K->sum($x); $K->finish(); if(!is_scalar($z)) $z = $z->toArray();
            $this->assertEquals(8,$z);
        }

        $x = $K->array([[2,3],[1,2]],NDArray::float32);
        $z = $K->sum($x,$axis=1); $K->finish();
        $this->assertEquals([5,3],$z->toArray());
        $this->assertEquals(NDArray::float32,$z->dtype());
        if($K->fp64()) {
            $x = $K->array([[2,3],[1,2]],NDArray::float64);
            $z = $K->sum($x,$axis=1); $K->finish();
            $this->assertEquals([5,3],$z->toArray());
            $this->assertEquals(NDArray::float64,$z->dtype());
        }

        /// ***** CAUTION *****
        /// OpenBLAS\Math::reduceSum not supports integer dtypes
        //$x = $mo->array([[2,3],[1,2]],NDArray::int32);
        //$this->assertEquals([5,3],$K->sum($x,$axis=1)->toArray());
        //$this->assertEquals(NDArray::float32,$K->sum($x,$axis=1)->dtype());
    }

    public function testMean()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $x = $K->array([[2,3],[1,2]]);
        $z = $K->mean($x,$axis=1); $K->finish();
        $this->assertEquals([2.5,1.5],$z->toArray());
    }

    public function testOneHot()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $x = $K->array([0,1,2,3,0,1,2,3]);
        $y = $K->oneHot($x,4); $K->finish();
        $this->assertEquals($y->toArray(),[
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1],
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1],
        ]);
    }

    public function testReLU()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $x = $K->array([-1.0,-0.5,0.0,0.5,1.0]);
        $y = $K->relu($x); $K->finish();
        $this->assertEquals([0.0,0.0,0.0,0.5,1.0],$y->toArray());
    }

    public function testSigmoid()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $x = $K->array([-1.0,-0.5,0.0,0.5,1.0]);
        $y = $K->sigmoid($x); $K->finish();

        $this->assertEquals([-1.0,-0.5,0.0,0.5,1.0],$x->toArray());
        $y = $y->toArray();
        $this->assertTrue($y[0]<0.5);
        $this->assertTrue($y[1]<0.5);
        $this->assertTrue($y[2]==0.5);
        $this->assertTrue($y[3]>0.5);
        $this->assertTrue($y[4]>0.5);
    }

    public function testSoftmax()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $fn = $K;
        $x = $K->array([-1.0,-0.5,0.0,0.5,1.0]);
        $y = $K->softmax($x); $K->finish();
        $y = $y->toArray();
        $this->assertTrue($y[0]>0.0);
        $this->assertTrue($y[0]<$y[1]);
        $this->assertTrue($y[1]<$y[2]);
        $this->assertTrue($y[2]<$y[3]);
        $this->assertTrue($y[3]<$y[4]);
        $this->assertTrue($y[4]<1.0);
        $this->assertTrue($fn->equalTest(1.0,$mo->sum($mo->array($y))));
        $single = $y;

        // batch mode
        $x = $K->array([
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
        ]);
        $y = $K->softmax($x); $K->finish();
        $this->assertEquals($single,$y[0]->toArray());
        $this->assertEquals($single,$y[1]->toArray());
        $this->assertEquals($single,$y[2]->toArray());
        $this->assertEquals($single,$y[3]->toArray());
        $this->assertEquals($single,$y[4]->toArray());

        $x = $K->array([
            [10,-10,2,8,-5],
            [10,-10,2,8,-5],
        ]);
        $softmax = $K->softmax($x); $K->finish();
        $this->assertLessThanOrEqual(1,$K->scalar($K->max($softmax)));
        $this->assertGreaterThanOrEqual(0,$K->scalar($K->max($softmax)));
        $sum = $K->sum($softmax,$axis=1); $K->finish(); $sum = $sum->toArray();
        $this->assertLessThan(0.0001,abs($sum[0]-1));
        $this->assertLessThan(0.0001,abs($sum[1]-1));
    }

    public function testMeanSquaredError()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $y = $K->array([-1.0,-0.5,0.0,0.5,1.0]);
        $t = $K->array([-1.0,-0.5,0.0,0.5,1.0]);
        $this->assertEquals(0.0,$K->scalar($K->meanSquaredError($y,$t)));

        $y = $K->array([-1.0,-0.5,0.1,0.5,1.0]);
        $t = $K->array([-1.0,-0.5,0.0,0.5,1.0]);
        $this->assertTrue(0.0<$K->scalar($K->meanSquaredError($y,$t)));
        $this->assertTrue(1.0>$K->scalar($K->meanSquaredError($y,$t)));

        $y = $K->array([-1.0,-0.5,0.0,0.5,1.0]);
        $t = $K->array([-1.0,-0.5,0.1,0.5,1.0]);
        $this->assertTrue(0.0<$K->scalar($K->meanSquaredError($y,$t)));
        $this->assertTrue(1.0>$K->scalar($K->meanSquaredError($y,$t)));
    }

    public function testSparseCategoricalCrossEntropy()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        // if test is label
        $y = $K->array([
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]);
        $t = $K->array([2,2]);
        $this->assertTrue($K->equalTest(
            0.0,$K->scalar($K->sparseCategoricalCrossEntropy($t,$y))));

        $y = $K->array([
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]);
        $t = $K->array([2]);
        $this->assertTrue($K->equalTest(
            0.0,$K->scalar($K->sparseCategoricalCrossEntropy($t,$y))));
    }

    public function testCategoricalCrossEntropy()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        // if test is label
        $y = $K->array([
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]);
        $t = $K->array([
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]);
        $this->assertTrue($K->equalTest(
            0.0,$K->categoricalCrossEntropy($t,$y)));

        $y = $K->array([
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]);
        $t = $K->array([
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]);
        $this->assertTrue($K->equalTest(
            0.0,$K->categoricalCrossEntropy($t,$y)));
    }

    public function testEqualArray()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $a = $K->array([1.0,2.0,3.0,4.0,5.0]);
        $b = $K->array([1.0,2.0,3.0,4.0,5.0]);
        $this->assertTrue($K->equalTest($a,$b));

        $b = $K->array([1.0,2.0,3.0,4.0,6.0]);
        $this->assertFalse($K->equalTest($a,$b));

        $b = $K->array([1.0,2.0,3.0,4.0, 5.0+1e-07 ]);
        $this->assertTrue($K->equalTest($a,$b));

        $b = $K->array([1.0,2.0,3.0,4.0, 5.0+9e-06 ]);
        $this->assertFalse($K->equalTest($a,$b));

        $b = $K->array([1.0,2.0,3.0,4.0, 5.0-1e-07 ]);
        $this->assertTrue($K->equalTest($a,$b));

        $b = $K->array([1.0,2.0,3.0,4.0, 5.0-9e-06 ]);
        $this->assertFalse($K->equalTest($a,$b));

        $b = $K->array([[1.0,2.0,3.0,4.0,5.0]]);
        $this->assertFalse($K->equalTest($a,$b));
    }

    public function testEqualNumeric()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);

        $this->assertTrue($K->equalTest(1,1));
        $this->assertFalse($K->equalTest(1,2));
        $this->assertTrue($K->equalTest(1, 1+9e-08));
        $this->assertTrue($K->equalTest(1, 1-9e-08));
        $this->assertFalse($K->equalTest(1, 1+9e-06));
        $this->assertFalse($K->equalTest(1, 1-9e-06));
    }

    public function testConv1d()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);

        $batches = 1;
        $im_w = 4;
        $channels = 3;
        $kernel_w = 3;
        $filters = 5;
        $stride_w = 1;
        $padding = null;
        $data_format = null;

        $inputs = $K->array($mo->arange(
            $batches*
            $im_w*
            $channels,
            null,null,
            NDArray::float32
        ))->reshape([
            $batches,
            $im_w,
            $channels
        ]);

        $kernel = $K->ones([
            $kernel_w,
            $channels,
            $filters
        ]);
        $bias = $K->zeros([
            $filters
        ]);

        $status = new \stdClass();

        $outputs = $K->conv1d(
            $status,
            $inputs,
            $kernel,
            $bias,
            $strides=[$stride_w],
            $padding,
            $data_format
        );
        $this->assertEquals(
            [$batches,
             $out_w=2,
             $filters],
            $outputs->shape()
        );

        $dOutputs = $K->ones($outputs->shape());
        $dKernel = $K->zerosLike($kernel);
        $dBias = $K->zerosLike($bias);
        $dInputs = $K->dConv1d(
            $status,
            $dOutputs,
            $dKernel,
            $dBias
        );

        $this->assertEquals(
            $inputs->shape(),
            $dInputs->shape()
            );
        $this->assertNotEquals(
            $dInputs->toArray(),
            $K->zerosLike($dInputs)->toArray()
            );
        $this->assertNotEquals(
            $dKernel->toArray(),
            $K->zerosLike($dKernel)->toArray()
            );
        $this->assertNotEquals(
            $dBias->toArray(),
            $K->zerosLike($dBias)->toArray()
            );
    }

    public function testPool1dMax()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);

        $batches = 1;
        $im_w = 4;
        $channels = 3;
        $pool_w = 2;
        #$stride_h = 1;
        #$stride_w = 1;
        $padding = null;
        $data_format = null;
        $pool_mode = null;

        $inputs = $K->array($mo->arange(
            $batches*
            $im_w*
            $channels,
            null,null,
            NDArray::float32
        ))->reshape([
            $batches,
            $im_w,
            $channels
        ]);

        $status = new \stdClass();

        $outputs = $K->pool1d(
            $status,
            $inputs,
            $poolSize=[$pool_w],
            $strides=null,
            $padding,
            $data_format,
            $pool_mode
        );
        $this->assertEquals(
            [$batches,
             $out_w=2,
             $channels],
            $outputs->shape()
        );
        $this->assertEquals([[
            [3,4,5],[9,10,11],
        ]],$outputs->toArray());
        $dOutputs = $K->ones($outputs->shape());
        $dInputs = $K->dPool1d(
            $status,
            $dOutputs
        );

        $this->assertEquals(
            $inputs->shape(),
            $dInputs->shape()
            );
        $this->assertNotEquals(
            $dInputs->toArray(),
            $K->zerosLike($dInputs)->toArray()
            );
    }

    public function testPool1dAvg()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);

        $batches = 1;
        $im_w = 4;
        $channels = 3;
        $pool_w = 2;
        #$stride_h = 1;
        #$stride_w = 1;
        $padding = null;
        $data_format = null;
        $pool_mode = 'avg';

        $inputs = $K->array($mo->arange(
            $batches*
            $im_w*
            $channels,
            null,null,
            NDArray::float32
        ))->reshape([
            $batches,
            $im_w,
            $channels
        ]);

        $status = new \stdClass();

        $outputs = $K->pool1d(
            $status,
            $inputs,
            $poolSize=[$pool_w],
            $strides=null,
            $padding,
            $data_format,
            $dilation_rate=null,
            $pool_mode
        );
        $this->assertEquals(
            [$batches,
             $out_w=2,
             $channels],
            $outputs->shape()
        );
        $this->assertEquals([[
            [1.5,2.5,3.5],[7.5,8.5,9.5],
        ]],$outputs->toArray());
        $dOutputs = $K->ones($outputs->shape());
        $dInputs = $K->dPool1d(
            $status,
            $dOutputs
        );

        $this->assertEquals(
            $inputs->shape(),
            $dInputs->shape()
            );
        $this->assertNotEquals(
            $dInputs->toArray(),
            $K->zerosLike($dInputs)->toArray()
            );
    }

    public function testConv2d()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);

        $batches = 1;
        $im_h = 4;
        $im_w = 4;
        $channels = 3;
        $kernel_h = 3;
        $kernel_w = 3;
        $filters = 5;
        $stride_h = 1;
        $stride_w = 1;
        $padding = null;
        $data_format = null;

        $inputs = $K->array($mo->arange(
            $batches*
            $im_h*$im_w*
            $channels,
            null,null,
            NDArray::float32
        ))->reshape([
            $batches,
            $im_h,
            $im_w,
            $channels
        ]);

        $kernel = $K->ones([
            $kernel_h,
            $kernel_w,
            $channels,
            $filters
        ]);
        $bias = $K->zeros([
            $filters
        ]);

        $status = new \stdClass();

        $outputs = $K->conv2d(
            $status,
            $inputs,
            $kernel,
            $bias,
            $strides=[$stride_h,$stride_w],
            $padding,
            $data_format
        );
        $this->assertEquals(
            [$batches,
             $out_h=2,
             $out_w=2,
             $filters],
            $outputs->shape()
        );

        $dOutputs = $K->ones($outputs->shape());
        $dKernel = $K->zerosLike($kernel);
        $dBias = $K->zerosLike($bias);
        $dInputs = $K->dConv2d(
            $status,
            $dOutputs,
            $dKernel,
            $dBias
        );

        $this->assertEquals(
            $inputs->shape(),
            $dInputs->shape()
            );
        $this->assertNotEquals(
            $dInputs->toArray(),
            $K->zerosLike($dInputs)->toArray()
            );
        $this->assertNotEquals(
            $dKernel->toArray(),
            $K->zerosLike($dKernel)->toArray()
            );
        $this->assertNotEquals(
            $dBias->toArray(),
            $K->zerosLike($dBias)->toArray()
            );
    }

    public function testPool2d()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);

        $batches = 1;
        $im_h = 4;
        $im_w = 4;
        $channels = 3;
        $pool_h = 2;
        $pool_w = 2;
        #$stride_h = 1;
        #$stride_w = 1;
        $padding = null;
        $data_format = null;
        $pool_mode = null;

        $inputs = $K->array($mo->arange(
            $batches*
            $im_h*$im_w*
            $channels,
            null,null,
            NDArray::float32
        ))->reshape([
            $batches,
            $im_h,
            $im_w,
            $channels
        ]);

        $status = new \stdClass();

        $outputs = $K->pool2d(
            $status,
            $inputs,
            $poolSize=[$pool_h,$pool_w],
            $strides=null,
            $padding,
            $data_format,
            $dilation_rate=null,
            $pool_mode
        );
        $this->assertEquals(
            [$batches,
             $out_h=2,
             $out_w=2,
             $channels],
            $outputs->shape()
        );
        $this->assertEquals([[
            [[15,16,17],[21,22,23]],
            [[39,40,41],[45,46,47]],
        ]],$outputs->toArray());
        $dOutputs = $K->ones($outputs->shape());
        $dInputs = $K->dPool2d(
            $status,
            $dOutputs
        );

        $this->assertEquals(
            $inputs->shape(),
            $dInputs->shape()
            );
        $this->assertNotEquals(
            $dInputs->toArray(),
            $K->zerosLike($dInputs)->toArray()
            );
    }

    public function testConv3d()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);

        $batches = 1;
        $im_d = 4;
        $im_h = 4;
        $im_w = 4;
        $channels = 3;
        $kernel_d = 3;
        $kernel_h = 3;
        $kernel_w = 3;
        $filters = 5;
        $stride_d = 1;
        $stride_h = 1;
        $stride_w = 1;
        $padding = null;
        $data_format = null;

        $inputs = $K->array($mo->arange(
            $batches*
            $im_d*$im_h*$im_w*
            $channels,
            null,null,
            NDArray::float32
        ))->reshape([
            $batches,
            $im_d,
            $im_h,
            $im_w,
            $channels
        ]);

        $kernel = $K->ones([
            $kernel_d,
            $kernel_h,
            $kernel_w,
            $channels,
            $filters
        ]);
        $bias = $K->zeros([
            $filters
        ]);

        $status = new \stdClass();

        $outputs = $K->conv3d(
            $status,
            $inputs,
            $kernel,
            $bias,
            $strides=[$stride_d,$stride_h,$stride_w],
            $padding,
            $data_format
        );
        $this->assertEquals(
            [$batches,
             $out_h=2,
             $out_h=2,
             $out_w=2,
             $filters],
            $outputs->shape()
        );

        $dOutputs = $K->ones($outputs->shape());
        $dKernel = $K->zerosLike($kernel);
        $dBias = $K->zerosLike($bias);
        $dInputs = $K->dConv3d(
            $status,
            $dOutputs,
            $dKernel,
            $dBias
        );

        $this->assertEquals(
            $inputs->shape(),
            $dInputs->shape()
            );
        $this->assertNotEquals(
            $dInputs->toArray(),
            $K->zerosLike($dInputs)->toArray()
            );
        $this->assertNotEquals(
            $dKernel->toArray(),
            $K->zerosLike($dKernel)->toArray()
            );
        $this->assertNotEquals(
            $dBias->toArray(),
            $K->zerosLike($dBias)->toArray()
            );
    }

    public function testPool3d()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);

        $batches = 1;
        $im_d = 4;
        $im_h = 4;
        $im_w = 4;
        $channels = 3;
        $pool_d = 2;
        $pool_h = 2;
        $pool_w = 2;
        #$stride_h = 1;
        #$stride_w = 1;
        $padding = null;
        $data_format = null;
        $pool_mode = null;

        $inputs = $K->array($mo->arange(
            $batches*
            $im_d*$im_h*$im_w*
            $channels,
            null,null,
            NDArray::float32
        ))->reshape([
            $batches,
            $im_d,
            $im_h,
            $im_w,
            $channels
        ]);

        $status = new \stdClass();

        $outputs = $K->pool3d(
            $status,
            $inputs,
            $poolSize=[$pool_d,$pool_h,$pool_w],
            $strides=null,
            $padding,
            $data_format,
            $dilation_rate=null,
            $pool_mode
        );
        $this->assertEquals(
            [$batches,
             $out_d=2,
             $out_h=2,
             $out_w=2,
             $channels],
            $outputs->shape()
        );
        /*
        $this->assertEquals([[
            [[15,16,17],[21,22,23]],
            [[39,40,41],[45,46,47]],
        ]],$outputs->toArray());
        */
        $dOutputs = $K->ones($outputs->shape());
        $dInputs = $K->dPool3d(
            $status,
            $dOutputs
        );

        $this->assertEquals(
            $inputs->shape(),
            $dInputs->shape()
            );
        $this->assertNotEquals(
            $dInputs->toArray(),
            $K->zerosLike($dInputs)->toArray()
            );
    }

    public function testGlorotNormal()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $w = $K->glorot_normal([16,4],[16,4]);
        #echo "--------\n";
        #foreach($w->toArray() as $array)
        #    echo '['.implode(',',array_map(function($a){return sprintf('%5.2f',$a);},$array))."],\n";
        #$this->assertLessThan(1.0/0.87962566103423978,abs($K->amax($w)));
        $max = $K->amax($w); if(!is_scalar($max)) $max = $max->toArray();
        $this->assertLessThan(1.8,abs($max));
        $min = $K->amin($w); if(!is_scalar($min)) $min = $min->toArray();
        $this->assertGreaterThan(1e-6,abs($min));

        $kernel = $K->glorot_normal([200,10],[200,10])->reshape([2000]);
        $min = $K->scalar($K->min($kernel));
        $max = $K->scalar($K->max($kernel));
        $kernel = $K->scale(1/($max-$min),$K->increment($kernel,-$min));
        $indices = $K->cast($K->scale(9.999,$kernel),NDArray::int32);
        $ones = $K->ones([2000,1]);
        $frequency = $K->zeros([10,1]);
        $K->scatterAdd($frequency,$indices,$ones);
        $plt = new Plot($this->getPlotConfig(),$mo);
        $x = $mo->arange(10,$min,($max-$min)/10,NDArray::float32);
        $plt->plot($x,$K->ndarray($frequency->reshape([10])));
        $plt->title('glorot_normal'.($K->accelerated()?':accel':''));
        $plt->show();
    }

    public function testGlorotUniform()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $w = $K->glorot_uniform([16,4],[16,4]);
        #echo "--------\n";
        #foreach($w->toArray() as $array)
        #    echo '['.implode(',',array_map(function($a){return sprintf('%5.2f',$a);},$array))."],\n";
        $max = $K->amax($w); if(!is_scalar($max)) $max = $max->toArray();
        $this->assertLessThan(1.0,abs($max));
        $min = $K->amin($w); if(!is_scalar($min)) $min = $min->toArray();
        $this->assertGreaterThan(1e-6,abs($min));

        $kernel = $K->glorot_uniform([200,10],[200,10])->reshape([2000]);
        $min = $K->scalar($K->min($kernel));
        $max = $K->scalar($K->max($kernel));
        $kernel = $K->scale(1/($max-$min),$K->increment($kernel,-$min));
        $indices = $K->cast($K->scale(9.999,$kernel),NDArray::int32);
        $ones = $K->ones([2000,1]);
        $frequency = $K->zeros([10,1]);
        $K->scatterAdd($frequency,$indices,$ones);
        $plt = new Plot($this->getPlotConfig(),$mo);
        $x = $mo->arange(10,$min,($max-$min)/10,NDArray::float32);
        $plt->plot($x,$K->ndarray($frequency->reshape([10])));
        $plt->title('glorot_uniform'.($K->accelerated()?':accel':''));
        $plt->show();
    }

    public function testHeUniform()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $w = $K->he_uniform([16,4],[16,4]);
        #echo "--------\n";
        #foreach($w->toArray() as $array)
        #    echo '['.implode(',',array_map(function($a){return sprintf('%5.2f',$a);},$array))."],\n";
        #$this->assertLessThan(1.0/0.87962566103423978,abs($K->amax($w)));
        $max = $K->amax($w); if(!is_scalar($max)) $max = $max->toArray();
        $this->assertLessThan(1.8,abs($max));
        $min = $K->amin($w); if(!is_scalar($min)) $min = $min->toArray();
        $this->assertGreaterThan(1e-6,abs($min));

        $kernel = $K->he_uniform([200,10],[200,10])->reshape([2000]);
        $min = $K->scalar($K->min($kernel));
        $max = $K->scalar($K->max($kernel));
        $kernel = $K->scale(1/($max-$min),$K->increment($kernel,-$min));
        $indices = $K->cast($K->scale(9.999,$kernel),NDArray::int32);
        $ones = $K->ones([2000,1]);
        $frequency = $K->zeros([10,1]);
        $K->scatterAdd($frequency,$indices,$ones);
        $plt = new Plot($this->getPlotConfig(),$mo);
        $x = $mo->arange(10,$min,($max-$min)/10,NDArray::float32);
        $plt->plot($x,$K->ndarray($frequency->reshape([10])));
        $plt->title('he_uniform'.($K->accelerated()?':accel':''));
        $plt->show();
    }

    public function testHeNormal()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $w = $K->he_normal([16,4],[16,4]);
        #echo "--------\n";
        #foreach($w->toArray() as $array)
        #    echo '['.implode(',',array_map(function($a){return sprintf('%5.2f',$a);},$array))."],\n";
        #$this->assertLessThan(1.0/0.87962566103423978,abs($K->amax($w)));
        $max = $K->amax($w); if(!is_scalar($max)) $max = $max->toArray();
        $this->assertLessThan(1.8,abs($max));
        $min = $K->amin($w); if(!is_scalar($min)) $min = $min->toArray();
        $this->assertGreaterThan(1e-6,abs($min));

        $kernel = $K->he_normal([200,10],[200,10])->reshape([2000]);
        $min = $K->scalar($K->min($kernel));
        $max = $K->scalar($K->max($kernel));
        $kernel = $K->scale(1/($max-$min),$K->increment($kernel,-$min));
        $indices = $K->cast($K->scale(9.999,$kernel),NDArray::int32);
        $ones = $K->ones([2000,1]);
        $frequency = $K->zeros([10,1]);
        $K->scatterAdd($frequency,$indices,$ones);
        $plt = new Plot($this->getPlotConfig(),$mo);
        $x = $mo->arange(10,$min,($max-$min)/10,NDArray::float32);
        $plt->plot($x,$K->ndarray($frequency->reshape([10])));
        $plt->title('he_normal'.($K->accelerated()?':accel':''));
        $plt->show();
    }

    public function testOrthogonal()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $w = $K->orthogonal([16,4]);
        #echo "--------\n";
        #foreach($w->toArray() as $array)
        #    echo '['.implode(',',array_map(function($a){return sprintf('%5.2f',$a);},$array))."],\n";
        $max = $K->amax($w); if(!is_scalar($max)) $max = $max->toArray();
        $this->assertLessThan(1.0,abs($max));
        $min = $K->amin($w); if(!is_scalar($min)) $min = $min->toArray();
        //$this->assertGreaterThan(1e-6,abs($min));

        $bins = 10;
        $m = 200;//200;
        $n = 10;//10;
        $kernel = $K->orthogonal([$m,$n],[$m,$n])->reshape([$m*$n]);
        $min = $K->scalar($K->min($kernel));
        $max = $K->scalar($K->max($kernel));
        //var_dump([$min,$max]);
        $kernel = $K->scale(1/($max-$min),$K->increment($kernel,-$min));
        $indices = $K->cast($K->scale($bins-0.001,$kernel),NDArray::int32);
        $ones = $K->ones([$m*$n,1]);
        $frequency = $K->zeros([$bins,1]);
        $K->scatterAdd($frequency,$indices,$ones);
        $plt = new Plot($this->getPlotConfig(),$mo);
        $x = $mo->arange($bins,$min,($max-$min)/$bins,NDArray::float32);
        $plt->plot($x,$K->ndarray($frequency->reshape([$bins])));
        $plt->title('orthogonal'.($K->accelerated()?':accel':''));
        $plt->show();
    }

}
