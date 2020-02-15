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
    public function testGetInitializer()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);

        $initializer = $backend->getInitializer('relu_normal');
        $this->assertInstanceof(
            'Rindow\NeuralNetworks\Backend\RindowBlas\Backend',
            $initializer[0]
        );
        $this->assertEquals('initializer_relu',$initializer[1]);
        $this->assertTrue(is_callable($initializer));
        $kernel = $initializer([2,3]);
        $this->assertInstanceof('Interop\Polite\Math\Matrix\NDArray',$kernel);
        $this->assertEquals([2,3],$kernel->shape());
        $this->assertEquals(NDArray::float32,$kernel->dtype());


        $initializer = $backend->getInitializer('sigmoid_normal');
        $this->assertInstanceof(
            'Rindow\NeuralNetworks\Backend\RindowBlas\Backend',
            $initializer[0]
        );
        $this->assertEquals('initializer_sigmoid',$initializer[1]);
        $this->assertTrue(is_callable($initializer));
        $kernel = $initializer([2,3]);
        $this->assertInstanceof('Interop\Polite\Math\Matrix\NDArray',$kernel);
        $this->assertEquals([2,3],$kernel->shape());
        $this->assertEquals(NDArray::float32,$kernel->dtype());


        $initializer = $backend->getInitializer('zeros');
        $this->assertInstanceof(
            'Rindow\NeuralNetworks\Backend\RindowBlas\Backend',
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
            'Rindow\NeuralNetworks\Backend\RindowBlas\Backend',
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
        $mo = new MatrixOperator();
        $backend = new Backend($mo);

        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Unsupported initializer: boo');
        $initializer = $backend->getInitializer('boo');
    }

    public function testZeros()
    {
        $mo = new MatrixOperator();
        $K = new Backend($mo);
        $this->assertEquals([[0,0]],$K->zeros([1,2])->toArray());
    }

    public function testOnes()
    {
        $mo = new MatrixOperator();
        $K = new Backend($mo);
        $this->assertEquals([[1,1]],$K->ones([1,2])->toArray());
    }

    public function testZeroLike()
    {
        $mo = new MatrixOperator();
        $K = new Backend($mo);
        $x = $mo->ones([1,2]);
        $this->assertEquals([[0,0]],$K->zerosLike($x)->toArray());
    }

    public function testOnesLike()
    {
        $mo = new MatrixOperator();
        $K = new Backend($mo);
        $x = $mo->zeros([1,2]);
        $this->assertEquals([[1,1]],$K->onesLike($x)->toArray());
    }

    public function testCast()
    {
        $mo = new MatrixOperator();
        $K = new Backend($mo);

        $x = $mo->ones([1,2],$dtype=NDArray::int32);
        $this->assertEquals(NDArray::int32,$x->dtype());
        $this->assertEquals(NDArray::float32,$K->cast($x,NDArray::float32)->dtype());
        $this->assertTrue(is_float($K->cast($x,NDArray::float32)[0][0]));
        $this->assertEquals(NDArray::bool,$K->cast($x,NDArray::bool)->dtype());
        $this->assertTrue(is_bool($K->cast($x,NDArray::bool)[0][0]));
        $this->assertEquals(true,$K->cast($x,NDArray::bool)[0][0]);

        $x = $mo->ones([1,2],$dtype=NDArray::float32);
        $this->assertEquals(NDArray::float32,$x->dtype());
        $this->assertEquals(NDArray::int32,$K->cast($x,NDArray::int32)->dtype());
        $this->assertTrue(is_int($K->cast($x,NDArray::int32)[0][0]));
        $this->assertEquals(NDArray::bool,$K->cast($x,NDArray::bool)->dtype());
        $this->assertTrue(is_bool($K->cast($x,NDArray::bool)[0][0]));
        $this->assertEquals(true,$K->cast($x,NDArray::bool)[0][0]);

        $x = $mo->array([[true,false]],$dtype=NDArray::bool);
        $this->assertEquals(NDArray::bool,$x->dtype());
        $this->assertEquals(NDArray::int32,$K->cast($x,NDArray::int32)->dtype());
        $this->assertTrue(is_int($K->cast($x,NDArray::int32)[0][0]));
        $this->assertEquals(1,$K->cast($x,NDArray::int32)[0][0]);
        $this->assertEquals(0,$K->cast($x,NDArray::int32)[0][1]);
        $this->assertEquals(NDArray::float32,$K->cast($x,NDArray::float32)->dtype());
        $this->assertTrue(is_float($K->cast($x,NDArray::float32)[0][0]));
        $this->assertEquals(1.0,$K->cast($x,NDArray::float32)[0][0]);
        $this->assertEquals(0.0,$K->cast($x,NDArray::float32)[0][1]);
    }

    public function testAdd()
    {
        $mo = new MatrixOperator();
        $K = new Backend($mo);
        $x = $mo->array([2,3]);
        $y = $mo->array([10,20]);
        $this->assertEquals([12,23],$K->add($x,$y)->toArray());
    }

    public function testSub()
    {
        $mo = new MatrixOperator();
        $K = new Backend($mo);
        $x = $mo->array([10,20]);
        $y = $mo->array([2,3]);
        $this->assertEquals([8,17],$K->sub($x,$y)->toArray());
    }

    public function testMul()
    {
        $mo = new MatrixOperator();
        $K = new Backend($mo);
        $x = $mo->array([10,20]);
        $y = $mo->array([2,3]);
        $this->assertEquals([20,60],$K->mul($x,$y)->toArray());
    }

    public function testDiv()
    {
        $mo = new MatrixOperator();
        $K = new Backend($mo);
        $x = $mo->array([10,21]);
        $y = $mo->array([2,3]);
        $this->assertEquals([5,7],$K->div($x,$y)->toArray());
    }

    public function testUpdate_add()
    {
        $mo = new MatrixOperator();
        $K = new Backend($mo);
        $x = $mo->array([10,20]);
        $y = $mo->array([2,3]);
        $this->assertEquals([12,23],$K->update_add($x,$y)->toArray());
        $this->assertEquals([12,23],$x->toArray());
    }

    public function testUpdate_sub()
    {
        $mo = new MatrixOperator();
        $K = new Backend($mo);
        $x = $mo->array([10,20]);
        $y = $mo->array([2,3]);
        $this->assertEquals([8,17],$K->update_sub($x,$y)->toArray());
        $this->assertEquals([8,17],$x->toArray());
    }

    public function testScale()
    {
        $mo = new MatrixOperator();
        $K = new Backend($mo);
        $x = $mo->array([2,3]);
        $this->assertEquals([4,6],$K->scale(2,$x)->toArray());
    }

    public function testPow()
    {
        $mo = new MatrixOperator();
        $K = new Backend($mo);
        $x = $mo->array([2,3]);
        $this->assertEquals([3,2],$K->scale(6,$K->pow($x,-1))->toArray());
    }

    public function testSqrt()
    {
        $mo = new MatrixOperator();
        $K = new Backend($mo);
        $x = $mo->array([4,9]);
        $this->assertEquals([2,3],$K->sqrt($x)->toArray());
    }

    public function testGreater()
    {
        $mo = new MatrixOperator();
        $K = new Backend($mo);
        $x = $mo->array([4,9]);
        $this->assertEquals([0,1],$K->greater($x,5)->toArray());
    }

    public function testEqual()
    {
        $mo = new MatrixOperator();
        $K = new Backend($mo);
        $x = $mo->array([4,9],NDArray::float32);
        $y = $mo->array([9,9],NDArray::float32);
        $this->assertEquals([0,1],$K->equal($x,$y)->toArray());
        $this->assertEquals(NDArray::float32,$K->equal($x,$y)->dtype());

        $x = $mo->array([4,9],NDArray::int32);
        $y = $mo->array([9,9],NDArray::int32);
        $this->assertEquals([0,1],$K->equal($x,$y)->toArray());
        $this->assertEquals(NDArray::int32,$K->equal($x,$y)->dtype());
    }

    public function testSum()
    {
        $mo = new MatrixOperator();
        $K = new Backend($mo);

        $x = $mo->array([[true,false],[true,true]],NDArray::bool);
        $this->assertEquals(3,$K->sum($x));
        $x = $mo->array([[2,3],[1,2]],NDArray::int8);
        $this->assertEquals(8,$K->sum($x));
        $x = $mo->array([[2,3],[1,2]],NDArray::uint8);
        $this->assertEquals(8,$K->sum($x));
        $x = $mo->array([[2,3],[1,2]],NDArray::int16);
        $this->assertEquals(8,$K->sum($x));
        $x = $mo->array([[2,3],[1,2]],NDArray::uint16);
        $this->assertEquals(8,$K->sum($x));
        $x = $mo->array([[2,3],[1,2]],NDArray::int32);
        $this->assertEquals(8,$K->sum($x));
        $x = $mo->array([[2,3],[1,2]],NDArray::uint32);
        $this->assertEquals(8,$K->sum($x));
        $x = $mo->array([[2,3],[1,2]],NDArray::int64);
        $this->assertEquals(8,$K->sum($x));
        $x = $mo->array([[2,3],[1,2]],NDArray::uint64);
        $this->assertEquals(8,$K->sum($x));
        $x = $mo->array([[2,3],[1,2]],NDArray::float32);
        $this->assertEquals(8,$K->sum($x));
        $x = $mo->array([[2,3],[1,2]],NDArray::float64);
        $this->assertEquals(8,$K->sum($x));

        $x = $mo->array([[2,3],[1,2]],NDArray::float32);
        $this->assertEquals([5,3],$K->sum($x,$axis=1)->toArray());
        $this->assertEquals(NDArray::float32,$K->sum($x,$axis=1)->dtype());
        $x = $mo->array([[2,3],[1,2]],NDArray::float64);
        $this->assertEquals([5,3],$K->sum($x,$axis=1)->toArray());
        $this->assertEquals(NDArray::float64,$K->sum($x,$axis=1)->dtype());

        /// ***** CAUTION *****
        /// OpenBLAS\Math::reduceSum not supports integer dtypes
        //$x = $mo->array([[2,3],[1,2]],NDArray::int32);
        //$this->assertEquals([5,3],$K->sum($x,$axis=1)->toArray());
        //$this->assertEquals(NDArray::float32,$K->sum($x,$axis=1)->dtype());
    }

    public function testMean()
    {
        $mo = new MatrixOperator();
        $K = new Backend($mo);
        $x = $mo->array([[2,3],[1,2]]);
        $this->assertEquals([2.5,1.5],$K->mean($x,$axis=1)->toArray());
    }

    public function testOneHot()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $x = $mo->array([0,1,2,3,0,1,2,3]);
        $y = $backend->oneHot($x,4);
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

    public function testRelu()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $x = $mo->array([-1.0,-0.5,0.0,0.5,1.0]);
        $y = $backend->relu($x);
        $this->assertTrue($y[0]==0.0);
        $this->assertTrue($y[1]==0.0);
        $this->assertTrue($y[2]==0.0);
        $this->assertTrue($y[3]==0.5);
        $this->assertTrue($y[4]==1.0);
    }

    public function testSigmoid()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $x = $mo->array([-1.0,-0.5,0.0,0.5,1.0]);
        $y = $backend->sigmoid($x);

        $this->assertEquals([-1.0,-0.5,0.0,0.5,1.0],$x->toArray());
        $this->assertTrue($y[0]<0.5);
        $this->assertTrue($y[1]<0.5);
        $this->assertTrue($y[2]==0.5);
        $this->assertTrue($y[3]>0.5);
        $this->assertTrue($y[4]>0.5);
    }

    public function testSoftmax()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $fn = $backend;
        $x = $mo->array([-1.0,-0.5,0.0,0.5,1.0]);
        $y = $backend->softmax($x);
        $this->assertTrue($y[0]>0.0);
        $this->assertTrue($y[0]<$y[1]);
        $this->assertTrue($y[1]<$y[2]);
        $this->assertTrue($y[2]<$y[3]);
        $this->assertTrue($y[3]<$y[4]);
        $this->assertTrue($y[4]<1.0);
        $this->assertTrue($fn->equalTest(1.0,$mo->sum($y)));
        $single = $y->toArray();

        // batch mode
        $x = $mo->array([
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
        ]);
        $y = $backend->softmax($x);
        $this->assertEquals($single,$y[0]->toArray());
        $this->assertEquals($single,$y[1]->toArray());
        $this->assertEquals($single,$y[2]->toArray());
        $this->assertEquals($single,$y[3]->toArray());
        $this->assertEquals($single,$y[4]->toArray());

        $x = $mo->array([
            [10,-10,2,8,-5],
            [10,-10,2,8,-5],
        ]);
        $softmax = $backend->softmax($x);
        $this->assertLessThanOrEqual(1,$mo->max($softmax));
        $this->assertGreaterThanOrEqual(0,$mo->max($softmax));
        $sum = $mo->sum($softmax,$axis=1)->toArray();
        $this->assertLessThan(0.0001,abs($sum[0]-1));
        $this->assertLessThan(0.0001,abs($sum[1]-1));
    }

    public function testMeanSquaredError()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $y = $mo->array([-1.0,-0.5,0.0,0.5,1.0]);
        $t = $mo->array([-1.0,-0.5,0.0,0.5,1.0]);
        $this->assertEquals(0.0,$backend->meanSquaredError($y,$t));

        $y = $mo->array([-1.0,-0.5,0.1,0.5,1.0]);
        $t = $mo->array([-1.0,-0.5,0.0,0.5,1.0]);
        $this->assertTrue(0.0<$backend->meanSquaredError($y,$t));
        $this->assertTrue(1.0>$backend->meanSquaredError($y,$t));

        $y = $mo->array([-1.0,-0.5,0.0,0.5,1.0]);
        $t = $mo->array([-1.0,-0.5,0.1,0.5,1.0]);
        $this->assertTrue(0.0<$backend->meanSquaredError($y,$t));
        $this->assertTrue(1.0>$backend->meanSquaredError($y,$t));
    }

    public function testSparseCategoricalCrossEntropy()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $fn = $backend;
        // if test is label
        $y = $mo->array([
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]);
        $t = $mo->array([2,2]);
        $this->assertTrue($fn->equalTest(
            0.0,$backend->sparseCategoricalCrossEntropy($t,$y)));

        $y = $mo->array([
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]);
        $t = $mo->array([2]);
        $this->assertTrue($fn->equalTest(
            0.0,$backend->sparseCategoricalCrossEntropy($t,$y)));
    }

    public function testCategoricalCrossEntropy()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $fn = $backend;
        // if test is label
        $y = $mo->array([
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]);
        $t = $mo->array([
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]);
        $this->assertTrue($fn->equalTest(
            0.0,$backend->categoricalCrossEntropy($t,$y)));

        $y = $mo->array([
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]);
        $t = $mo->array([
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]);
        $this->assertTrue($fn->equalTest(
            0.0,$backend->categoricalCrossEntropy($t,$y)));
    }

/*
    public function testPlotSigmoid()
    {
        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);
        $plt = new Plot();
        $x = $mo->arange(100,-5.0,0.1);
        $y = $nn->fn()->sigmoid($x);
        $plt->plot($x,$y);
        $plt->show();
    }

    public function testPlotRelu()
    {
        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);
        $plt = new Plot();
        $x = $mo->arange(100,-5.0,0.1);
        $y = $nn->fn()->relu($x);
        $plt->plot($x,$y);
        $plt->show();
    }

    public function testPlotSoftmax()
    {
        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);
        $plt = new Plot();
        $x = $mo->arange(100,-5.0,0.1);
        $y = $nn->fn()->softmax($x);
        $plt->plot($x,$y);
        $plt->show();
    }
*/
    public function testEqualArray()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $fn = $backend;
        $a = $mo->array([1.0,2.0,3.0,4.0,5.0]);
        $b = $mo->array([1.0,2.0,3.0,4.0,5.0]);
        $this->assertTrue($fn->equalTest($a,$b));

        $b = $mo->array([1.0,2.0,3.0,4.0,6.0]);
        $this->assertFalse($fn->equalTest($a,$b));

        $b = $mo->array([1.0,2.0,3.0,4.0, 5.0+1e-07 ]);
        $this->assertTrue($fn->equalTest($a,$b));

        $b = $mo->array([1.0,2.0,3.0,4.0, 5.0+9e-06 ]);
        $this->assertFalse($fn->equalTest($a,$b));

        $b = $mo->array([1.0,2.0,3.0,4.0, 5.0-1e-07 ]);
        $this->assertTrue($fn->equalTest($a,$b));

        $b = $mo->array([1.0,2.0,3.0,4.0, 5.0-9e-06 ]);
        $this->assertFalse($fn->equalTest($a,$b));

        $b = $mo->array([[1.0,2.0,3.0,4.0,5.0]]);
        $this->assertFalse($fn->equalTest($a,$b));
    }

    public function testEqualNumeric()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $fn = $backend;

        $this->assertTrue($fn->equalTest(1,1));
        $this->assertFalse($fn->equalTest(1,2));
        $this->assertTrue($fn->equalTest(1, 1+9e-08));
        $this->assertTrue($fn->equalTest(1, 1-9e-08));
        $this->assertFalse($fn->equalTest(1, 1+9e-06));
        $this->assertFalse($fn->equalTest(1, 1-9e-06));
    }
}
