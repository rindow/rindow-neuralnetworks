<?php
namespace RindowTest\NeuralNetworks\Backend\RindowBlas\BackendTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use SplFixedArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use InvalidArgumentException;

class BackendTest extends TestCase
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
            'renderer.skipRunViewer' => getenv('PLOT_RENDERER_SKIP') ? true : false,
        ];
    }

    public function testGetInitializer()
    {
        $mo = $this->newMatrixOperator();
        $backend = $this->newBackend($mo);
        $backendClassName = get_class($backend);

        $initializer = $backend->getInitializer('he_normal',-0.1,0.1);
        $this->assertTrue(is_callable($initializer));
        $kernel = $initializer([2,3]);
        $this->assertInstanceof('Interop\Polite\Math\Matrix\NDArray',$kernel);
        $this->assertEquals([2,3],$kernel->shape());
        $this->assertEquals(NDArray::float32,$kernel->dtype());


        $initializer = $backend->getInitializer('glorot_normal');
        $this->assertTrue(is_callable($initializer));
        $kernel = $initializer([2,3],[3,6]);
        $this->assertInstanceof('Interop\Polite\Math\Matrix\NDArray',$kernel);
        $this->assertEquals([2,3],$kernel->shape());
        $this->assertEquals(NDArray::float32,$kernel->dtype());


        $initializer = $backend->getInitializer('zeros');
        $this->assertTrue(is_callable($initializer));
        $kernel = $initializer([2]);
        $this->assertInstanceof('Interop\Polite\Math\Matrix\NDArray',$kernel);
        $this->assertEquals([2],$kernel->shape());
        $this->assertEquals(NDArray::float32,$kernel->dtype());

        $initializer = $backend->getInitializer('ones');
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

        // broadcast x <- y
        $x = $K->zeros([2,3]);
        $y = $K->array([1,2,3]);
        $z = $K->add($x,$y); $K->finish();
        $this->assertEquals([
            [1,2,3],
            [1,2,3],
        ],$z->toArray());

        // broadcast x -> y
        $x = $K->array([1,2,3]);
        $y = $K->zeros([2,3]);
        $z = $K->add($x,$y); $K->finish();
        $this->assertEquals([
            [1,2,3],
            [1,2,3],
        ],$z->toArray());

        // transpose
        $x = $K->zeros([2,3]);
        $y = $K->array([1,2]);
        $z = $K->add($x,$y,trans:true); $K->finish();
        $this->assertEquals([
            [1,1,1],
            [2,2,2],
        ],$z->toArray());
    }

    public function testSub()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);

        $x = $K->array([10,20]);
        $y = $K->array([2,3]);
        $z = $K->sub($x,$y); $K->finish();
        $this->assertEquals([8,17],$z->toArray());

        // broadcast x <- y
        $x = $K->zeros([2,3]);
        $y = $K->array([1,2,3]);
        $z = $K->sub($x,$y); $K->finish();
        $this->assertEquals([
            [-1,-2,-3],
            [-1,-2,-3],
        ],$z->toArray());

        // broadcast x -> y
        $x = $K->array([1,2,3]);
        $y = $K->zeros([2,3]);
        $z = $K->sub($x,$y); $K->finish();
        $this->assertEquals([
            [1,2,3],
            [1,2,3],
        ],$z->toArray());

        // transpose
        $x = $K->zeros([2,3]);
        $y = $K->array([1,2]);
        $z = $K->sub($x,$y,trans:true); $K->finish();
        $this->assertEquals([
            [-1,-1,-1],
            [-2,-2,-2],
        ],$z->toArray());
    }

    public function testMul()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $x = $K->array([10,20]);
        $y = $K->array([2,3]);
        $z = $K->mul($x,$y); $K->finish();
        $this->assertEquals([20,60],$z->toArray());
        
        // broadcast x <- y
        $x = $K->ones([2,3]);
        $y = $K->array([1,2,3]);
        $z = $K->mul($x,$y); $K->finish();
        $this->assertEquals([
            [1,2,3],
            [1,2,3],
        ],$z->toArray());

        // broadcast x -> y
        $x = $K->array([1,2,3]);
        $y = $K->ones([2,3]);
        $z = $K->mul($x,$y); $K->finish();
        $this->assertEquals([
            [1,2,3],
            [1,2,3],
        ],$z->toArray());

        // transpose
        $x = $K->ones([2,3]);
        $y = $K->array([1,2]);
        $z = $K->mul($x,$y,trans:true); $K->finish();
        $this->assertEquals([
            [1,1,1],
            [2,2,2],
        ],$z->toArray());
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

        // broadcast x <- y
        $x = $K->fill([2,3],6);
        $y = $K->array([1,2,3]);
        $z = $K->div($x,$y); $K->finish();
        $this->assertEquals([
            [6,3,2],
            [6,3,2],
        ],$z->toArray());

        // broadcast x -> y
        $x = $K->array([2,4,6]);
        $y = $K->fill([2,3],2);
        $z = $K->div($x,$y); $K->finish();
        $this->assertEquals([
            [1,2,3],
            [1,2,3],
        ],$z->toArray());

        // transpose
        $x = $K->fill([2,3],6);
        $y = $K->array([1,2]);
        $z = $K->div($x,$y,trans:true); $K->finish();
        $this->assertEquals([
            [6,6,6],
            [3,3,3],
        ],$z->toArray());
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
        $this->assertEquals([2,3],$y->toArray());

        $x = $K->array([[10,20],[30,40]]);
        $y = $K->array([2,3]);
        $z = $K->update_add($x,$y); $K->finish();
        $this->assertEquals([[12,23],[32,43]],$z->toArray());
        $this->assertEquals([[12,23],[32,43]],$z->toArray());
        $this->assertEquals([2,3],$y->toArray());
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
        $la = $K->primaryLA();
        $x = $K->array([2,3]);
        $z = $K->scale(6,$K->pow($x,-1)); $K->finish();
        //$this->assertEquals([3,2],$z->toArray());
        $z = $la->toNDArray($z);
        $this->assertTrue($mo->la()->isclose(
            $mo->array([3,2]),
            $z,
        ));
    }

    public function testSqrt()
    {
        $mo = $this->newMatrixOperator();
        $la = $mo->la();
        $K = $this->newBackend($mo);
        $x = $K->array([4,9]);
        $z = $K->sqrt($x); $K->finish();
        $this->assertTrue($la->isclose($la->array([2,3]),$K->ndarray($z)));
    }

    public function testAbs()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $x = $K->array([-2,-1, 0, 1, 2]);
        $z = $K->abs($x); $K->finish();
        $this->assertEquals([-2,-1, 0, 1, 2],$x->toArray());
        $this->assertEquals([ 2, 1, 0, 1, 2],$z->toArray());
    }

    public function testSign()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $x = $K->array([-2,-1, 0, 1, 2]);
        $z = $K->sign($x); $K->finish();
        $this->assertEquals([-2,-1, 0, 1, 2],$x->toArray());
        $this->assertEquals([-1,-1, 0, 1, 1],$z->toArray());
    }

    public function testMaximum()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $x = $K->array([4,5,9]);
        $z = $K->maximum($x,5); $K->finish();
        $this->assertEquals([4,5,9],$x->toArray());
        $this->assertEquals([5,5,9],$z->toArray());
    }

    public function testMinimum()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $x = $K->array([4,5,9]);
        $z = $K->minimum($x,5); $K->finish();
        $this->assertEquals([4,5,9],$x->toArray());
        $this->assertEquals([4,5,5],$z->toArray());
    }

    public function testGreater()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $x = $K->array([4,5,9]);
        $z = $K->greater($x,5); $K->finish();
        $this->assertEquals([4,5,9],$x->toArray());
        $this->assertEquals([0,0,1],$z->toArray());
    }

    public function testGreaterEqual()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $x = $K->array([4,5,9]);
        $z = $K->greaterEqual($x,5); $K->finish();
        $this->assertEquals([4,5,9],$x->toArray());
        $this->assertEquals([0,1,1],$z->toArray());
    }

    public function testLess()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $x = $K->array([4,5,9]);
        $z = $K->less($x,5); $K->finish();
        $this->assertEquals([4,5,9],$x->toArray());
        $this->assertEquals([1,0,0],$z->toArray());
    }

    public function testLessEqual()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $x = $K->array([4,5,9]);
        $z = $K->lessEqual($x,5); $K->finish();
        $this->assertEquals([4,5,9],$x->toArray());
        $this->assertEquals([1,1,0],$z->toArray());
    }

    public function testEqual()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $x = $K->array([4,9],NDArray::float32);
        $y = $K->array([9,9],NDArray::float32);
        $x_backup = $K->copy($x); $K->finish();
        $y_backup = $K->copy($y); $K->finish();

        $z = $K->equal($x,$y); $K->finish();
        $this->assertEquals([0,1],$z->toArray());
        $this->assertEquals(NDArray::float32,$z->dtype());
        $this->assertEquals($x_backup->toArray(),$x->toArray());
        $this->assertEquals($y_backup->toArray(),$y->toArray());

        $x = $K->array([4,9],NDArray::int32);
        $y = $K->array([9,9],NDArray::int32);
        $x_backup = $K->copy($x); $K->finish();
        $y_backup = $K->copy($y); $K->finish();

        $z = $K->equal($x,$y); $K->finish();
        $this->assertEquals([0,1],$z->toArray());
        $this->assertEquals(NDArray::int32,$z->dtype());
        $this->assertEquals($x_backup->toArray(),$x->toArray());
        $this->assertEquals($y_backup->toArray(),$y->toArray());
    }

    public function testNotEqual()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $x = $K->array([4,9],NDArray::float32);
        $y = $K->array([9,9],NDArray::float32);
        $x_backup = $K->copy($x); $K->finish();
        $y_backup = $K->copy($y); $K->finish();

        $z = $K->notEqual($x,$y); $K->finish();
        $this->assertEquals([1,0],$z->toArray());
        $this->assertEquals(NDArray::float32,$z->dtype());
        $this->assertEquals($x_backup->toArray(),$x->toArray());
        $this->assertEquals($y_backup->toArray(),$y->toArray());

        $x = $K->array([4,9],NDArray::int32);
        $y = $K->array([9,9],NDArray::int32);
        $x_backup = $K->copy($x); $K->finish();
        $y_backup = $K->copy($y); $K->finish();

        $z = $K->notEqual($x,$y); $K->finish();
        $this->assertEquals([1,0],$z->toArray());
        $this->assertEquals(NDArray::int32,$z->dtype());
        $this->assertEquals($x_backup->toArray(),$x->toArray());
        $this->assertEquals($y_backup->toArray(),$y->toArray());
    }

    public function testNot()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $x = $K->array([4,0],NDArray::float32);
        $x_backup = $K->copy($x); $K->finish();

        $z = $K->not($x); $K->finish();
        $this->assertEquals([0,1],$z->toArray());
        $this->assertEquals(NDArray::float32,$z->dtype());
        $this->assertEquals($x_backup->toArray(),$x->toArray());

        $x = $K->array([0,9],NDArray::int32);
        $x_backup = $K->copy($x); $K->finish();

        $z = $K->not($x); $K->finish();
        $this->assertEquals([1,0],$z->toArray());
        $this->assertEquals(NDArray::int32,$z->dtype());
        $this->assertEquals($x_backup->toArray(),$x->toArray());
    }

    public function testSin()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $la = $K->localLA();
        $x = $K->array([M_PI,M_PI*0.5],NDArray::float32);
        $y = $K->ndarray($K->sin($x));
        $this->assertTrue($la->isclose(
            $mo->array([0,1]),$y));
    }

    public function testCos()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $la = $K->localLA();
        $x = $K->array([M_PI,M_PI*0.5],NDArray::float32);
        $y = $K->ndarray($K->cos($x));
        $this->assertTrue($la->isclose(
            $mo->array([-1,0]),$y));
    }

    public function testTan()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $la = $K->localLA();
        $x = $K->array([0,M_PI*0.25],NDArray::float32);
        $y = $K->ndarray($K->tan($x));
        $this->assertTrue($la->isclose(
            $mo->array([0,1]),$y));
    }

    public function testTanh()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $la = $K->localLA();
        $x = $K->array([-10,0,10],NDArray::float32);
        $y = $K->ndarray($K->tanh($x));
        $this->assertTrue($la->isclose(
            $mo->array([-1,0,1]),$y));
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
        $z = $K->sum($x,axis:1); $K->finish();
        $this->assertEquals([5,3],$z->toArray());
        $this->assertEquals(NDArray::float32,$z->dtype());
        if($K->fp64()) {
            $x = $K->array([[2,3],[1,2]],NDArray::float64);
            $z = $K->sum($x,axis:1); $K->finish();
            $this->assertEquals([5,3],$z->toArray());
            $this->assertEquals(NDArray::float64,$z->dtype());
        }

        /// ***** CAUTION *****
        /// OpenBLAS\Math::reduceSum not supports integer dtypes
        //$x = $mo->array([[2,3],[1,2]],NDArray::int32);
        //$this->assertEquals([5,3],$K->sum($x,axis:1)->toArray());
        //$this->assertEquals(NDArray::float32,$K->sum($x,axis:1)->dtype());
    }

    public function testMean()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $x = $K->array([[2,3],[1,2]]);
        $z = $K->mean($x,axis:1); $K->finish();
        $this->assertEquals([2.5,1.5],$z->toArray());
        $z = $K->mean($x); $K->finish();
        $this->assertLessThan(1e-5,abs(((2+3+1+2)/4)-$z));
    }

    public function testStd()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $x = $K->array([[2,3],[1,4]]);
        $z = $K->std($x,$axis=1); $K->finish();
        $this->assertEquals([0.5,1.5],$z->toArray());
        $z = $K->std($x); $K->finish();
        $this->assertLessThan(1e-5, 1.1180339-$z);
    }

    public function testmax()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $x = $K->array([[2,3],[1,2]]);
        $z = $K->max($x,axis:1); $K->finish();
        $this->assertEquals([3,2],$z->toArray());
        $z = $K->max($x); $K->finish();
        $z = $K->scalar($z);
        $this->assertLessThan(1e-5,abs(3-$z));
    }

    public function testmin()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $x = $K->array([[2,3],[1,2]]);
        $z = $K->min($x,axis:1); $K->finish();
        $this->assertEquals([2,1],$z->toArray());
        $z = $K->min($x); $K->finish();
        $z = $K->scalar($z);
        $this->assertLessThan(1e-5,abs(1-$z));
    }

    public function testamax()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $x = $K->array([[2,-3],[1,-2]]);
        //$z = $K->amax($x,axis:1); $K->finish();
        //$this->assertEquals([2,1],$z->toArray());
        $z = $K->amax($x); $K->finish();
        $z = $K->scalar($z);
        $this->assertLessThan(1e-5,abs(3-$z));
    }

    public function testamin()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $x = $K->array([[-2,3],[-1,2]]);
        //$z = $K->amax($x,axis:1); $K->finish();
        //$this->assertEquals([2,1],$z->toArray());
        $z = $K->amin($x); $K->finish();
        $z = $K->scalar($z);
        $this->assertLessThan(1e-5,abs(1-$z));
    }

    public function testargmax()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $x = $K->array([[2,3],[1,2]]);
        $z = $K->argMax($x,axis:1); $K->finish();
        $this->assertEquals([1,1],$z->toArray());
        $z = $K->argMax($x); $K->finish();
        $z = $K->scalar($z);
        $this->assertEquals(1,$z);
    }

    public function testargmin()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $x = $K->array([[2,3],[1,2]]);
        if(!$K->accelerated()) {
            $z = $K->argMin($x,axis:1); $K->finish();
            $this->assertEquals([0,0],$z->toArray());
        }
        $z = $K->argMin($x); $K->finish();
        $z = $K->scalar($z);
        $this->assertEquals(2,$z);
    }

    public function testNrm2()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $x = $K->array([[1,2],[3,4]],dtype:NDArray::float32);
        $nrm2 = sqrt(1+2**2+3**2+4**2);
        $z = $K->nrm2($x); $K->finish();
        $z = $K->scalar($z);
        $this->assertLessThan(0.00001,abs($nrm2-$z));
    }

    public function testRand()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $z = $K->rand([2,2]); $K->finish();
        $this->assertLessThan(1.0,$K->scalar($K->max($z)));
        $this->assertGreaterThan(0.0,$K->scalar($K->min($z)));
    }

    public function testRandomSequence()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $z = $K->randomSequence(8,size:4); $K->finish();
        $this->assertLessThan(8,$K->scalar($K->max($z)));
        $this->assertGreaterThanOrEqual(0,$K->scalar($K->min($z)));
    }

    public function testDot()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $x = $K->array([[1,2,3],[4,5,6]],dtype:NDArray::float32);
        $y = $K->array([[10,20,30],[40,50,60]],dtype:NDArray::float32);
        $z = $K->dot($x,$y); $K->finish();
        $z = $K->scalar($z);
        $this->assertEquals(1*10+2*20+3*30+4*40+5*50+6*60,$z);
    }

    public function testGemm()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $A = $K->array([
            [1,2,3],
            [4,5,6],
            [7,8,9],
        ]);
        $B = $K->array([
            [1,0,0],
            [0,1,0],
            [0,0,1],
        ]);
        $C = $K->array([
            [100,200,300],
            [400,500,600],
            [700,800,900],
        ]);
        $Z = $K->gemm($A,$B,beta:1,c:$C); $K->finish();
        $this->assertEquals([
            [101,202,303],
            [404,505,606],
            [707,808,909],
        ],$Z->toArray());
    }

    public function testBatchGemm()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $A = $K->array([
            [1,2,3],
            [4,5,6],
            [7,8,9],
        ]);
        $B = $K->array([
            [1,0,0],
            [0,1,0],
            [0,0,1],
        ]);
        $C = $K->array(
            [100,200,300],
        );
        $Z = $K->batch_gemm($A,$B,beta:1,c:$C); $K->finish();
        $this->assertEquals([
            [101,202,303],
            [104,205,306],
            [107,208,309],
        ],$Z->toArray());
    }

    public function testMatmul()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $A = $K->array([
            [1,2,3],
            [4,5,6],
            [7,8,9],
        ]);
        $B = $K->array([
            [1,0,0],
            [0,1,0],
            [0,0,1],
        ]);
        $Z = $K->matmul($A,$B); $K->finish();
        $this->assertEquals([
            [1,2,3],
            [4,5,6],
            [7,8,9],
        ],$Z->toArray());
    }

    public function testexpandDims()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);

        $x = $K->zeros([2,3]);
        $this->assertEquals([2,3],$x->shape());
        $this->assertEquals([1,2,3],$K->expandDims($x,axis:0)->shape());
        $this->assertEquals([2,1,3],$K->expandDims($x,axis:1)->shape());

        $x = $K->zeros([]);
        $this->assertEquals([],$x->shape());
        $this->assertEquals(1,$x->size());
        $this->assertEquals(1,count($x->buffer()));
        $this->assertEquals([1],$K->expandDims($x,axis:0)->shape());
        $this->assertEquals([1],$K->expandDims($x,axis:-1)->shape());
    }

    public function testSqueeze()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $x = $K->zeros([1,2,1,3,1]);
        $this->assertEquals([1,2,1,3,1],$x->shape());
        $this->assertEquals([2,3],$K->squeeze($x)->shape());
        $this->assertEquals([2,1,3,1],$K->squeeze($x,axis:0)->shape());
        $this->assertEquals([1,2,3,1],$K->squeeze($x,axis:2)->shape());
        $this->assertEquals([1,2,1,3],$K->squeeze($x,axis:4)->shape());
        $this->assertEquals([1,2,1,3],$K->squeeze($x,axis:-1)->shape());
        $this->assertEquals([1,2,3,1],$K->squeeze($x,axis:-3)->shape());
        $this->assertEquals([2,1,3,1],$K->squeeze($x,axis:-5)->shape());
    }

    public function testGather()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);

        // 1D by 1D
        $x = $K->array([3,2,1,1],dtype:NDArray::int32);
        $a = $K->array([10,11,12,13,14,15,16,17,18,19]);
        $b = $K->gather($a,$x); $K->finish();
        $this->assertEquals([4],$b->shape()); // replace axis0
        $this->assertEquals([13,12,11,11],$b->toArray());

        // axis = 0
        // 1D indices
        $x = $K->array([3,2,1],dtype:NDArray::int32);
        $a = $K->array([
            [ 0, 0, 3],
            [ 0, 0, 4],
            [ 0, 2, 0],
            [ 1, 0, 0]]);
        $b = $K->gather($a,$x,axis:0);  $K->finish();
        $this->assertEquals([3],$x->shape());
        $this->assertEquals([4,3],$a->shape());
        $this->assertEquals([3],$b->shape()); // reduction axis0
        $this->assertEquals([1,2,4],$b->toArray()); // reduction axis0
    }

    public function testScatterAdd()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);

        // 1D by 1D
        $x = $K->array([3,2,1,4],dtype:NDArray::int32); // Must not be duplicated
        $a = $K->array([13,12,11,14]);
        $b = $K->ones([10]);  $K->finish();
        $K->scatterAdd($b,$x,$a);  $K->finish();
        $this->assertEquals([4],$x->shape());
        $this->assertEquals([4],$a->shape());
        $this->assertEquals([10],$b->shape()); // replace axis0
        $trues = $K->array([1,12,13,14,15,1,1,1,1,1]);
        //echo $mo->toString($b,null,true)."\n";
        $this->assertEquals($trues->toArray(),$b->toArray());

        //
        // axis = 0
        //
        //  1D inputs
        $x = $K->array([3,2,0],dtype:NDArray::int32);
        $a = $K->array([1,2,3],dtype:NDArray::float32);
        $b = $K->ones([4,3]); $K->finish();
        $K->scatterAdd($b,$x,$a,axis:0); $K->finish();
        $this->assertEquals([3],$x->shape());
        $this->assertEquals([3],$a->shape());
        $this->assertEquals([4,3],$b->shape()); // insert axis0
        $trues = $K->array([
            [ 1, 1, 4],
            [ 1, 1, 1],
            [ 1, 3, 1],
            [ 2, 1, 1]]);
        $this->assertEquals($trues->toArray(),$b->toArray());
    }

    public function testGatherb()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);

        // 1D by 1D
        $a = $K->array([10,11,12,13,14,15,16,17,18,19]);
        $x = $K->array([3,2,1,1],dtype:NDArray::int32);
        $b = $K->gatherb($a,$x); $K->finish();
        $this->assertEquals([4],$b->shape()); // replace axis0
        $this->assertEquals([13,12,11,11],$b->toArray());

        // axis = 0
        // detailDepth = 2
        // indexDepth = 0
        // params:(numClass(4),inner(3))
        // indices:(inner(3))
        // 1D indices
        $a = $K->array([
            [ 0, 0, 3],
            [ 0, 0, 4],
            [ 0, 2, 0],
            [ 1, 0, 0]]);
        $x = $K->array([3,2,1],dtype:NDArray::int32);
        $b = $K->gatherb($a,$x,axis:0,detailDepth:2,indexDepth:0);  $K->finish();
        $this->assertEquals([4,3],$a->shape());
        $this->assertEquals([3],$x->shape());
        $this->assertEquals([3],$b->shape()); // reduction axis0
        $this->assertEquals([1,2,4],$b->toArray()); // reduction axis0
    }

    public function testScatterbAdd()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);

        // 1D by 1D
        $x = $K->array([3,2,1,4],dtype:NDArray::int32); // Must not be duplicated
        $a = $K->array([13,12,11,14]);
        $b = $K->ones([10]);  $K->finish();
        $shape = [10];
        $K->scatterbAdd($x,$a,$shape,outputs:$b);  $K->finish();
        $this->assertEquals([4],$x->shape());
        $this->assertEquals([4],$a->shape());
        $this->assertEquals([10],$b->shape()); // replace axis0
        $trues = $K->array([1,12,13,14,15,1,1,1,1,1]);
        //echo $mo->toString($b,null,true)."\n";
        $this->assertEquals($trues->toArray(),$b->toArray());

        //
        // axis = 0
        //
        //  1D inputs
        $x = $K->array([3,2,0],dtype:NDArray::int32);
        $a = $K->array([1,2,3],dtype:NDArray::float32);
        $b = $K->ones([4,3]); $K->finish();
        $shape = [4,3];
        $K->scatterbAdd($x,$a,$shape,axis:0,detailDepth:2,indexDepth:0,outputs:$b); $K->finish();
        $this->assertEquals([3],$x->shape());
        $this->assertEquals([3],$a->shape());
        $this->assertEquals([4,3],$b->shape()); // insert axis0
        $trues = $K->array([
            [ 1, 1, 4],
            [ 1, 1, 1],
            [ 1, 3, 1],
            [ 2, 1, 1]]);
        $this->assertEquals($trues->toArray(),$b->toArray());
    }

    public function testSlice()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);

        // 3D
        $x = $K->array([
            [[0,1,2],
             [3,4,5],
             [6,7,8],
             [9,10,11]],
            [[12,13,14],
             [15,16,17],
             [18,19,20],
             [21,22,23]],
        ]);
        $this->assertEquals(3,$x->ndim());
        $y = $K->slice(
            $x,
            begin:[0,1],
            size:[-1,2]
            );
        $K->finish();
        $this->assertEquals([
            [[3,4,5],
             [6,7,8],],
            [[15,16,17],
             [18,19,20],],
        ],$y->toArray());
    }

    public function testStick()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);

        $x = $K->array($mo->arange(12,null,null,dtype:NDArray::float32)->reshape([2,2,3]));
        $y = $K->array($mo->zeros([2,4,3]));
        $K->stick(
            $x,
            $y,
            begin:[0,1],
            size:[-1,2]
            );
        $K->finish();
        $this->assertEquals([
            [[0,0,0],
             [0,1,2],
             [3,4,5],
             [0,0,0]],
            [[0,0,0],
             [6,7,8],
             [9,10,11],
             [0,0,0]],
        ],$y->toArray());
    }

    public function testStack()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);

        $a = $K->array($mo->arange(6,0,null,dtype:NDArray::float32)->reshape([2,3]));
        $b = $K->array($mo->arange(6,6,null,dtype:NDArray::float32)->reshape([2,3]));
        $y = $K->stack(
            [$a,$b],
            axis:0
            );
        $K->finish();

        $this->assertEquals([
            [[0,1,2],
             [3,4,5]],
            [[6,7,8],
             [9,10,11]],
        ],$y->toArray());
    }

    public function testConcat()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);

        $a = $K->array($mo->arange(6,$start=0,null,dtype:NDArray::float32)->reshape([3,2]));
        $b = $K->array($mo->arange(4,$start=6,null,dtype:NDArray::float32)->reshape([2,2]));
        $y = $K->concat(
            [$a,$b],
            axis:0
            );
        $K->finish();
        $this->assertEquals([
            [0,1],
            [2,3],
            [4,5],
            [6,7],
            [8,9],
        ],$y->toArray());
    }

    public function testSplit()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);

        $x = $K->array([
            [0,1],
            [2,3],
            [4,5],
            [6,7],
            [8,9],
        ]);
        $y = $K->split(
            $x,
            [3,2],
            axis:0
        );
        $K->finish();
        $a = $K->array($mo->arange(6,$start=0,null,dtype:NDArray::float32)->reshape([3,2]));
        $b = $K->array($mo->arange(4,$start=6,null,dtype:NDArray::float32)->reshape([2,2]));
        $this->assertCount(2,$y);
        $this->assertEquals($a->toArray(),$y[0]->toArray());
        $this->assertEquals($b->toArray(),$y[1]->toArray());
    }

    public function testRepeat()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);

        // Y := X (duplicate 2 times)
        $X = $K->array([
            [1,2,3],
            [4,5,6]
        ]);
        $Y = $K->repeat($X,$repeats=2,axis:1);
        $K->finish();
        $this->assertEquals([2,3],$X->shape());
        $this->assertEquals([2,2,3],$Y->shape());
        $this->assertEquals([
            [1,2,3],
            [4,5,6]
        ],$X->toArray());
        $this->assertEquals([
            [[1,2,3],[1,2,3]],
            [[4,5,6],[4,5,6]],
        ],$Y->toArray());
    }

    public function testOneHot()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $x = $K->array([0,1,2,3,0,1,2,3],dtype:NDArray::int32);
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

    public function testRandomUniformVariables()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);

        // float
        $x = $K->randomUniformVariables(
            shape:[20,30],
            low:-1.0,
            high:1.0);
        $K->finish();
        $y = $K->randomUniformVariables(
            shape:[20,30],
            low:-1,
            high:1);
        $K->finish();
        $this->assertEquals(
            NDArray::float32,$x->dtype());
        $this->assertNotEquals(
            $x->toArray(),
            $y->toArray());

        // int
        $x = $K->randomUniformVariables(
            shape:[20,30],
            low:-1,
            high:1,
            dtype:NDArray::int32
            );
        $K->finish();
        $y = $K->randomUniformVariables(
            shape:[20,30],
            low:-1,
            high:1,
            dtype:NDArray::int32);
        $K->finish();
        $this->assertEquals(
            NDArray::int32,$x->dtype());
        $this->assertNotEquals(
            $x->toArray(),
            $y->toArray());
    }

    public function testRandomNormalVariables()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);

        $x = $K->randomNormalVariables(
            shape:[20,30],
            mean:0.0,
            scale:1.0);
        $K->finish();
        $y = $K->randomNormalVariables(
            shape:[20,30],
            mean:0.0,
            scale:1.0);
        $K->finish();
        $this->assertEquals(
            NDArray::float32,$x->dtype());
        $this->assertNotEquals(
            $x->toArray(),
            $y->toArray());
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
        $ndy = $K->ndarray($y);
        $truesY = $mo->array([0.26894143223763,0.37754067778587,0.5,0.62245935201645,0.7310585975647]);
        $this->assertTrue($mo->la()->isclose($truesY,$ndy));

        // backward
        $dy = $K->onesLike($y);
        $dx = $K->dSigmoid($dy,$y); $K->finish();
        $nddx = $K->ndarray($dx);
        $truesDx = $mo->array([0.19661194086075,0.23500370979309,0.25,0.23500370979309,0.196611925959590]);
        $this->assertTrue($mo->la()->isclose($truesDx,$nddx));
    }

    public function testSoftmax0()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $fn = $K;
        $x = $K->array([-1.0,-0.5,0.0,0.5,1.0]);
        $y = $K->softmax($x); $K->finish();
        $ndy = $K->ndarray($y);
        $truesY = $mo->array([0.058012217283249,0.095645979046822,0.15769356489182,0.25999271869659,0.42865553498268]);
        $this->assertTrue($mo->la()->isclose($truesY,$ndy));
        $this->assertTrue($fn->equalTest(1.0,$mo->sum($ndy)));
        $dy = $K->onesLike($y);
        $dx = $K->dSoftmax($dy,$y); $K->finish();
        $nddx = $K->ndarray($dx);
        $truesDx = $mo->array([0,0,0,0,0]);
        $this->assertTrue($mo->la()->isclose($truesDx,$nddx));


        $single = $y->toArray();

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
        $sum = $K->sum($softmax,axis:1); $K->finish(); $sum = $sum->toArray();
        $this->assertLessThan(0.0001,abs($sum[0]-1));
        $this->assertLessThan(0.0001,abs($sum[1]-1));
    }

    public function testSoftmax1()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $fn = $K;
        $x = $K->array([
           [[[1., 2., 3., 4.],
             [1., 2., 3., 4.],
             [1., 2., 3., 4.]],
            [[1., 2., 3., 4.],
             [1., 2., 3., 4.],
             [1., 2., 3., 4.]]],
           [[[1., 2., 3., 4.],
             [1., 2., 3., 4.],
             [1., 2., 3., 4.]],
            [[1., 2., 3., 4.],
             [1., 2., 3., 4.],
             [1., 2., 3., 4.]]],
        ]);
        $salt = $mo->la()->range(array_product($x->shape()),dtype:NDArray::float32)
                ->reshape($x->shape());
        $salt = $K->array($salt);

        $softmax = $K->softmax($x);
        $y = $K->mul($salt,$softmax); $K->finish();
        //echo "softmax:".$mo->toString($y,format:'%10.7f',indent:true)."\n";
        //$ndy = $K->ndarray($y);
        //$truesY = $mo->array([0.058012217283249,0.095645979046822,0.15769356489182,0.25999271869659,0.42865553498268]);
        //$this->assertTrue($mo->la()->isclose($truesY,$ndy));
        //$this->assertTrue($fn->equalTest(1.0,$mo->sum($ndy)));
        $doutputs = $K->copy($salt);
        $dx = $K->dSoftmax($doutputs,$softmax); $K->finish();
        //echo "dSoftmax:".$mo->toString($dx,format:'%10.7f',indent:true)."\n";
        //$nddx = $K->ndarray($dx);
        //$truesDx = $mo->array([0,0,0,0,0]);
        //$this->assertTrue($mo->la()->isclose($truesDx,$nddx));
        $this->assertTrue(true);
    }

    public function testMeanSquaredError()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $reduction = 'sum';
        $y = $K->array([
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
        ]);
        $t = $K->array([
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
        ]);
        $loss = $K->meanSquaredError($y,$t,$reduction);
        $this->assertEquals(0.0,$K->scalar($loss));
        $dLoss = $K->onesLike($loss);
        $dy = $K->dMeanSquaredError($dLoss,$t,$y,$reduction);
        $trueDy = $mo->array([
            [0,0,0,0,0],
            [0,0,0,0,0],
        ]);
        $this->assertTrue($mo->la()->isclose($trueDy,$K->ndarray($dy)));

        $y = $K->array([-1.0,-0.5,0.1,0.5,1.0]);
        $t = $K->array([-1.0,-0.5,0.0,0.5,1.0]);
        $this->assertTrue(0.0<$K->scalar($K->meanSquaredError($y,$t)));
        $this->assertTrue(1.0>$K->scalar($K->meanSquaredError($y,$t)));

        $y = $K->array([-1.0,-0.5,0.0,0.5,1.0]);
        $t = $K->array([-1.0,-0.5,0.1,0.5,1.0]);
        $this->assertTrue(0.0<$K->scalar($K->meanSquaredError($y,$t)));
        $this->assertTrue(1.0>$K->scalar($K->meanSquaredError($y,$t)));

        // none reduction
        $reduction = 'none';
        $y = $K->array([
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
        ]);
        $t = $K->array([
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
        ]);
        $loss = $K->meanSquaredError($y,$t,$reduction);
        $trueLoss = $mo->array([0,0]);
        $this->assertTrue($mo->la()->isclose($trueLoss,$K->ndarray($loss)));
        $dLoss = $K->onesLike($loss);
        $dy = $K->dMeanSquaredError($dLoss,$t,$y,$reduction);
        $trueDy = $mo->array([
            [0,0,0,0,0],
            [0,0,0,0,0],
        ]);
        $this->assertTrue($mo->la()->isclose($trueDy,$K->ndarray($dy)));
    }

    public function testSparseCategoricalCrossEntropy()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $reduction = 'sum';
        // if test is label
        $y = $K->array([
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]);
        $t = $K->array([2,2],dtype:NDArray::int32);
        $loss = $K->sparseCategoricalCrossEntropy($t,$y,reduction:$reduction);
        $this->assertTrue($K->equalTest(0.0,$K->scalar($loss)));
        $dLoss = $K->zerosLike($loss);
        $dy = $K->dSparseCategoricalCrossEntropy($dLoss,$t,$y,reduction:$reduction);
        $trueDy = $mo->array([
            [0,0,0,0,0],
            [0,0,0,0,0],
        ]);
        $this->assertTrue($mo->la()->isclose($trueDy,$K->ndarray($dy)));

        $y = $K->array([
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]);
        $t = $K->array([2],dtype:NDArray::int32);
        $this->assertTrue($K->equalTest(
            0.0,$K->scalar($K->sparseCategoricalCrossEntropy($t,$y))));

        $reduction = 'none';
        // if test is label
        $y = $K->array([
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]);
        $t = $K->array([2,2],dtype:NDArray::int32);
        $loss = $K->sparseCategoricalCrossEntropy($t,$y,reduction:$reduction);
        $trueLoss = $mo->array([0,0]);
        $this->assertTrue($mo->la()->isclose($trueLoss,$K->ndarray($loss),atol:1e-6));
        $dLoss = $K->zerosLike($loss);
        $dy = $K->dSparseCategoricalCrossEntropy($dLoss,$t,$y,reduction:$reduction);
        $trueDy = $mo->array([
            [0,0,0,0,0],
            [0,0,0,0,0],
        ]);
        $this->assertTrue($mo->la()->isclose($trueDy,$K->ndarray($dy)));

    }

    public function testCategoricalCrossEntropy()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        // if test is label
        $reduction = 'sum';
        $y = $K->array([
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]);
        $t = $K->array([
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]);
        $loss = $K->categoricalCrossEntropy($t,$y,reduction:$reduction);
        $this->assertTrue($K->equalTest(0.0,$K->scalar($loss)));
        $dLoss = $K->zerosLike($loss);
        $dy = $K->dCategoricalCrossEntropy($dLoss,$t,$y,reduction:$reduction);
        $trueDy = $mo->array([
            [0,0,0,0,0],
            [0,0,0,0,0],
        ]);
        $this->assertTrue($mo->la()->isclose($trueDy,$K->ndarray($dy)));

        $y = $K->array([
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]);
        $t = $K->array([
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]);
        $this->assertTrue($K->equalTest(
            0.0,$K->scalar($K->categoricalCrossEntropy($t,$y))));

        
        $reduction = 'none';
        $y = $K->array([
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]);
        $t = $K->array([
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]);
        $loss = $K->categoricalCrossEntropy($t,$y,reduction:$reduction);
        $trueLoss = $mo->array([0,0]);
        $this->assertTrue($mo->la()->isclose($trueLoss,$K->ndarray($loss),atol:1e-6));
        $dLoss = $K->zerosLike($loss);
        $dy = $K->dCategoricalCrossEntropy($dLoss,$t,$y,reduction:$reduction);
        $trueDy = $mo->array([
            [0,0,0,0,0],
            [0,0,0,0,0],
        ]);
        $this->assertTrue($mo->la()->isclose($trueDy,$K->ndarray($dy)));
    }

    public function testBinaryCrossEntropy()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        // if test is label
        $reduction = 'sum';
        $y = $K->array([
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]);
        $t = $K->array([
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]);
        $loss = $K->binaryCrossEntropy($t,$y,reduction:$reduction);
        $this->assertTrue($K->equalTest(0.0,$K->scalar($loss)));
        $dLoss = $K->zerosLike($loss);
        $dy = $K->dBinaryCrossEntropy($dLoss,$t,$y,reduction:$reduction);
        $trueDy = $mo->array([
            [0,0,0,0,0],
            [0,0,0,0,0],
        ]);
        $this->assertTrue($mo->la()->isclose($trueDy,$K->ndarray($dy)));

        
        $reduction = 'none';
        $y = $K->array([
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]);
        $t = $K->array([
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]);
        $loss = $K->binaryCrossEntropy($t,$y,reduction:$reduction);
        $trueLoss = $mo->array([0,0]);
        $this->assertTrue($mo->la()->isclose($trueLoss,$K->ndarray($loss),atol:1e-6));
        $dLoss = $K->zerosLike($loss);
        $dy = $K->dBinaryCrossEntropy($dLoss,$t,$y,reduction:$reduction);
        $trueDy = $mo->array([
            [0,0,0,0,0],
            [0,0,0,0,0],
        ]);
        $this->assertTrue($mo->la()->isclose($trueDy,$K->ndarray($dy)));
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

    public function testMaskingNormal()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);

        //
        // Same shape (implicit batchDims:0, axis:0)
        //echo "==== Same shape (implicit batchDims:0, axis:batchDims)\n";
        //
        // X:(2,3)
        // A:(2,3)
        // outer:(),bro:(),inner:(2,3),bro2:()
        // m=2*3,n=1,k=1,len=1  ==translate==> m=1,n=1,k=2*3,len=1
        $X = $K->array([[true,false,true],[false,true,false]], dtype:NDArray::bool);
        $A = $K->array([[1,10,100],[-1,-10,-100]]);
        $A = $K->masking($X,$A);
        $this->assertEquals(
            [[true,false,true],[false,true,false]]
        ,$X->toArray());
        $this->assertEquals(
            [[1, 0,100],[0,-10, 0]]
        ,$A->toArray());

        //
        // broadcast to details
        //echo "==== broadcast to details\n";
        //
        // X:(2,3  )
        // A:(2,3,4)
        // outer:(2,3),bro:(4),inner:(),bro2:()
        // m=2*3,n=4,k=1,len=1  ==translate==> m=1,n=1,k=2*3,len=4
        $X = $K->array([[true,false,true],[false,true,false]], dtype:NDArray::bool);
        $A = $K->array([
            [[1,11,111,1111],[2,12,122,1222],[-3,13,133,1333]],
            [[1,21,121,1211],[2,22,222,2222],[-3,23,233,2333]]
        ]);
        $A = $K->masking($X,$A,batchDims:$X->ndim(),axis:$A->ndim());
        $this->assertEquals(
            [[true,false,true],[false,true,false]]
        ,$X->toArray());
        $this->assertEquals([
            [[1,11,111,1111],[0, 0,  0,   0],[-3,13,133,1333]],
            [[0, 0,  0,   0],[2,22,222,2222],[0, 0,  0,   0]]
        ],$A->toArray());

        //
        // broadcast to details
        //echo "==== broadcast to details for implicit\n";
        //
        // X:(2,3  )
        // A:(2,3,4)
        // outer:(2,3),bro:(4),inner:(),bro2:()
        // m=2*3,n=4,k=1,len=1  ==translate==> m=1,n=1,k=2*3,len=4
        $X = $K->array([[true,false,true],[false,true,false]], dtype:NDArray::bool);
        $A = $K->array([
            [[1,11,111,1111],[2,12,122,1222],[-3,13,133,1333]],
            [[1,21,121,1211],[2,22,222,2222],[-3,23,233,2333]]
        ]);
        $A = $K->masking($X,$A,batchDims:$X->ndim());
        $this->assertEquals(
            [[true,false,true],[false,true,false]]
        ,$X->toArray());
        $this->assertEquals([
            [[1,11,111,1111],[0, 0,  0,   0],[-3,13,133,1333]],
            [[0, 0,  0,   0],[2,22,222,2222],[0, 0,  0,   0]]
        ],$A->toArray());

        //
        // broadcast with gap
        //echo "==== broadcast with gap\n";
        //
        // X:(2,  3)
        // A:(2,4,3)
        // outer:(2),bro:(4),inner:(3),bro2:()
        // m=2,n=4,k=3,len=1
        $X = $K->array([
            [true,false,true],
            [false,true,false]
        ], dtype:NDArray::bool);
        $A = $K->array([
            [[1,11,111],[2,12,112],[-3,13,113],[-4,14,114]],
            [[1,21,211],[2,22,222],[-3,23,223],[-4,24,224]],
        ]);
        $A = $K->masking($X,$A,batchDims:1,axis:2);
        $this->assertEquals([
            [true,false,true],
            [false,true,false]
        ],$X->toArray());
        $this->assertEquals([
            [[1, 0,111],[2, 0,112],[-3, 0,113],[-4, 0,114]],
            [[0,21,  0],[0,22,  0],[ 0,23,  0],[ 0,24,  0]],
        ],$A->toArray());

        //
        // broadcast to rows (implicit batchDims:0)
        //echo "==== broadcast to rows (implicit batchDims:0)\n";
        //
        // X:(  2,3)
        // A:(4,2,3)
        // outer:(),bro:(4),inner:(2,3),bro2:()
        // m=1,n=2,k=2*3,len=1
        $X = $K->array([[true,false,true],[false,true,false]],dtype:NDArray::bool);
        $A = $K->array([
            [[1,11,111],[2,12,112]],
            [[1,21,211],[2,22,222]],
            [[1,31,311],[2,32,322]],
            [[1,41,411],[2,42,422]],
        ]);
        $A = $K->masking($X,$A,axis:1);
        $this->assertEquals(
            [[true,false,true],[false,true,false]]
        ,$X->toArray());
        $this->assertEquals([
            [[1, 0,111],[0,12,  0]],
            [[1, 0,211],[0,22,  0]],
            [[1, 0,311],[0,32,  0]],
            [[1, 0,411],[0,42,  0]],
        ],$A->toArray());

        //
        // broadcast to rows (implicit axis:batchDims)
        //echo "==== broadcast to rows (implicit axis:batchDims)\n";
        //
        // X:(2,3)
        // A:(2,3)
        // outer:(2),bro:(),inner:(3),bro2:()
        // m=2,n=1,k=3,len=1  ==translate==> m=1,n=1,k=2*3,len=1
        $X = $K->array([[true,false,true],[false,true,false]], dtype:NDArray::bool);
        $A = $K->array([[1,10,100],[-1,-10,-100]]);
        $A = $K->masking($X,$A,batchDims:1);
        $this->assertEquals(
            [[true,false,true],[false,true,false]]
        ,$X->toArray());
        $this->assertEquals(
            [[1, 0,100],[0,-10, 0]]
        ,$A->toArray());

        //
        // broadcast to rows (implicit batchDims:0, minus axis)
        //echo "==== broadcast to rows (implicit batchDims:0, minus axis)\n";
        //
        // X:(  2,3)
        // A:(4,2,3)
        // outer:(),bro:(4),inner:(2,3),bro2:()
        // m=1,n=4,k=2*3,len=1
        $X = $K->array([[true,false,true],[false,true,false]],dtype:NDArray::bool);
        $A = $K->array([
            [[1,11,111],[2,12,112]],
            [[1,21,211],[2,22,222]],
            [[1,31,311],[2,32,322]],
            [[1,41,411],[2,42,422]],
        ]);
        $A = $K->masking($X,$A,axis:-$X->ndim());
        $this->assertEquals(
            [[true,false,true],[false,true,false]]
        ,$X->toArray());
        $this->assertEquals([
            [[1, 0,111],[0,12,  0]],
            [[1, 0,211],[0,22,  0]],
            [[1, 0,311],[0,32,  0]],
            [[1, 0,411],[0,42,  0]],
        ],$A->toArray());

        //
        // broadcast with gap and implicit len
        //echo "==== broadcast with gap implicit len\n";
        //
        // X:(2,  3  )
        // A:(2,4,3,2)
        // outer:(2),bro:(4),inner:(3),bro2:(2)
        // m=2,n=4,k=3,len=2
        $X = $K->array([
            [true,false,true],
            [false,true,false]
        ], dtype:NDArray::bool);
        $A = $K->array([
            [[[1,-1],[11,-11],[111,-111]],
             [[2,-2],[12,-12],[112,-112]],
             [[-3,3],[13,-13],[113,-113]],
             [[-4,4],[14,-14],[114,-114]]],
            [[[1,-1],[21,-21],[211,-211]],
             [[2,-2],[22,-22],[222,-222]],
             [[-3,3],[23,-23],[223,-223]],
             [[-4,4],[24,-24],[224,-224]]],
        ]);
        $A = $K->masking($X,$A,batchDims:1,axis:2);
        $this->assertEquals([
            [true,false,true],
            [false,true,false]
        ],$X->toArray());
        $this->assertEquals([
            [[[1,-1],[ 0,  0],[111,-111]],
             [[2,-2],[ 0,  0],[112,-112]],
             [[-3,3],[ 0,  0],[113,-113]],
             [[-4,4],[ 0,  0],[114,-114]]],
            [[[0, 0],[21,-21],[  0,   0]],
             [[0, 0],[22,-22],[  0,   0]],
             [[0, 0],[23,-23],[  0,   0]],
             [[0, 0],[24,-24],[  0,   0]]],
        ],$A->toArray());

        //
        // broadcast to rows (implicit batchDims:0, implicit len)
        //echo "==== broadcast to rows (implicit batchDims:0, axis=1, implicit len)\n";
        //
        // X:(  2  )
        // A:(4,2,3)
        // outer:(),bro:(4),inner:(2),bro2:(3)
        // m=1,n=4,k=2,len=3
        $X = $K->array([true,false],dtype:NDArray::bool);
        $A = $K->array([
            [[1,11,111],[2,12,112]],
            [[1,21,211],[2,22,222]],
            [[1,31,311],[2,32,322]],
            [[1,41,411],[2,42,422]],
        ]);
        $A = $K->masking($X,$A,axis:1);
        $this->assertEquals([true,false],$X->toArray());
        $this->assertEquals([
            [[1,11,111],[0, 0,  0]],
            [[1,21,211],[0, 0,  0]],
            [[1,31,311],[0, 0,  0]],
            [[1,41,411],[0, 0,  0]],
        ],$A->toArray());

        //
        // fill -9999
        //
        //echo "==== fill -9999\n";
        // X:(2,3)
        // A:(2,3)
        // outer:(),bro:(),inner:(2,3),bro2:()
        // m=2*3,n=1,k=1  ==translate==> m=1,n=1,k=2*3
        $X = $K->array([[true,false,true],[false,true,false]], dtype:NDArray::bool);
        $A = $K->array([[1,10,100],[-1,-10,-100]]);
        $A = $K->masking($X,$A,fill:-9999);
        $this->assertEquals(
            [[true,false,true],[false,true,false]]
        ,$X->toArray());
        $this->assertEquals(
            [[1, -9999,100],[-9999,-10, -9999]]
        ,$A->toArray());
    }

    public function testMaskingAddMode()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);

        //
        // Same shape (implicit batchDims:0, axis:0)
        //echo "==== Same shape (implicit batchDims:0, axis:batchDims)\n";
        //
        // X:(2,3)
        // A:(2,3)
        // outer:(),bro:(),inner:(2,3),bro2:()
        // m=2*3,n=1,k=1,len=1  ==translate==> m=1,n=1,k=2*3,len=1
        $X = $K->array([[true,false,true],[false,true,false]], dtype:NDArray::bool);
        $A = $K->array([[1,10,100],[-1,-10,-100]]);
        $A = $K->masking($X,$A, fill:-1000, mode:1); // 0:set mode,  1:add mode
        $this->assertEquals(
            [[true,false,true],[false,true,false]]
        ,$X->toArray());
        $this->assertEquals(
            [[1, -990,100],[-1001,-10, -1100]]
        ,$A->toArray());
    }

    public function testUpdateMaskingAddMode()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);

        // broadcast to details
        //echo "==== broadcast to details\n";
        //
        // X:(2,3  )
        // A:(2,3,4)
        // outer:(2,3),bro:(4),inner:(),bro2:()
        // m=2*3,n=4,k=1,len=1  ==translate==> m=1,n=1,k=2*3,len=4
        $X = $K->array([[true,false,true],[false,true,false]], dtype:NDArray::bool);
        $A = $K->array([
            [[1,11,111,1111],[2,12,122,1222],[-3,13,133,1333]],
            [[1,21,121,1211],[2,22,222,2222],[-3,23,233,2333]]
        ]);
        $A = $K->masking($X,$A,fill:10000,mode:1, batchDims:$X->ndim(),axis:$A->ndim());
        $this->assertEquals(
            [[true,false,true],[false,true,false]]
        ,$X->toArray());
        $this->assertEquals([
            [[1,11,111,1111],[10002,10012,10122,11222],[-3,13,133,1333]],
            [[10001,10021,10121,11211],[2,22,222,2222],[9997,10023,10233,12333]]
        ],$A->toArray());
    }

}
