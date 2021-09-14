<?php
namespace RindowTest\NeuralNetworks\Loss\SparseCategoricalCrossEntropyTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Loss\SparseCategoricalCrossEntropy;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Plot\Plot;

class Test extends TestCase
{
    public function newMatrixOperator()
    {
        return new MatrixOperator();
    }

    public function newNN($mo)
    {
        $builder = new NeuralNetworks($mo);
        return [$builder,$builder->backend()];
    }

    public function verifyGradient($mo, $K, $function, NDArray $t, NDArray $x,$fromLogits=null)
    {
        $f = function($x) use ($mo,$K,$function,$t,$fromLogits){
            $x = $K->array($x);
            $t = $K->array($t);
            //if($fromLogits) {
            //    $x = $function->forward($x,true);
            //}
            $l = $function->forward($t,$x);
            return $mo->array([$l]);
        };
        $grads = $mo->la()->numericalGradient(1e-3,$f,$x);
        $outputs = $function->forward($K->array($t),$K->array($x));
        $dInputs = $function->backward([$K->array(1.0)]);
        $dInputs = $dInputs[0];
        $dInputs = $K->ndarray($dInputs);
//echo "\n";
//echo "grads=".$mo->toString($grads[0],'%5.3f',true)."\n\n";
//echo "dInputs=".$mo->toString($dInputs,'%5.3f',true)."\n\n";
//echo $mo->asum($mo->op($grads[0],'-',$dInputs))."\n";
        return $mo->la()->isclose($grads[0],$dInputs,null,1e-4);
    }

    public function getPlotConfig()
    {
        return [
            'renderer.skipCleaning' => true,
            'renderer.skipRunViewer' => getenv('TRAVIS_PHP_VERSION') ? true : false,
        ];
    }

    public function testGraph()
    {
        $mo = $this->newMatrixOperator();
        [$nn,$K] = $this->newNN($mo);
        $plt = new Plot($this->getPlotConfig(),$mo);
        $loss = $nn->losses()->SparseCategoricalCrossEntropy();
        $x = [[0.1,0.9],[0.3,0.7],[0.5,0.5],[0.7,0.3],[0.9,0.1]];
        $t = [0,0,0,0,0];
        $y = [];
        foreach($x as $k => $xx) {
            $tt = $t[$k];
            $y[] = $loss->forward($K->array([$tt]),$K->array([$xx]));
        }
        #$plt->plot($mo->array($y));
        $plt->show();
        $this->assertTrue(true);
    }

    public function testBuilder()
    {
        $mo = $this->newMatrixOperator();
        [$nn,$K] = $this->newNN($mo);
        $this->assertInstanceof(
            'Rindow\NeuralNetworks\Loss\SparseCategoricalCrossEntropy',
            $nn->losses()->SparseCategoricalCrossEntropy());
    }

    public function testDefault()
    {
        $mo = $this->newMatrixOperator();
        [$nn,$K] = $this->newNN($mo);
        $func = $nn->losses()->SparseCategoricalCrossEntropy();

        $x = $mo->array([
            [0.00001, 0.00001 , 0.99998],
            [0.99998, 0.00001 , 0.00001],
        ]);
        $t = $mo->array([2, 0],NDArray::int32);
        $copyx = $mo->copy($x);
        $copyt = $mo->copy($t);
        $t = $K->array($t);
        $x = $K->array($x);
        $loss = $func->forward($t,$x);
        $tt = $K->ndarray($t);
        $tx = $K->ndarray($x);
        $this->assertLessThan(0.001,abs($loss));
        $this->assertEquals($copyx->toArray(),$tx->toArray());
        $this->assertEquals($copyt->toArray(),$tt->toArray());
        $dx = $func->backward([$K->array(1.0)]);
        $dx = $dx[0];
        $dx = $K->ndarray($dx);
        //$this->assertLessThan(0.00001,abs($mo->sum($dx)));
        //$this->assertLessThan(0.001,abs(1-$dx[0][0])/6);
        $this->assertEquals($copyx->toArray(),$tx->toArray());
        $this->assertEquals($copyt->toArray(),$tt->toArray());

        $accuracy = $func->accuracy($t,$x);

        $x = $mo->array([
            [0.00001, 0.00001 , 0.99998],
            [0.99998, 0.00001 , 0.00001],
        ]);
        $t = $mo->array([1, 1]);
        $t = $K->array($t);
        $x = $K->array($x);
        $loss = $func->forward($t,$x);
        $this->assertGreaterThan(10,abs($loss));
        $dx = $func->backward([$K->array(1.0)]);
        $dx = $dx[0];
        $dx = $K->ndarray($dx);

        $x = $mo->array([
            [0.00001, 0.20000 , 0.79998],
            [0.79998, 0.00001 , 0.20000],
        ]);
        $t = $mo->array([2, 2]);
        $this->assertTrue(
            $this->verifyGradient($mo,$K,$func,$t,$x));
    }

    public function testFromLogits()
    {
        $mo = $this->newMatrixOperator();
        [$nn,$K] = $this->newNN($mo);
        $func = $nn->losses()->SparseCategoricalCrossEntropy();
        $func->setFromLogits(true);

        $x = $mo->array([
            [-10.0, -10.0 , 10.0],
            [ 10.0, -10.0 ,-10.0],
        ]);
        $t = $mo->array([2, 0],NDArray::int32);
        $copyx = $mo->copy($x);
        $copyt = $mo->copy($t);
        $t = $K->array($t);
        $x = $K->array($x);
        //$y = $func->forward($x,true);
        $loss = $func->forward($t,$x);
        $tt = $K->ndarray($t);
        $tx = $K->ndarray($x);
        $this->assertLessThan(0.00001,abs($loss));
        $this->assertEquals($copyx->toArray(),$tx->toArray());
        $this->assertEquals($copyt->toArray(),$tt->toArray());

        $dx = $func->backward([$K->array(1.0)]);
        $dx =  $dx[0];
        $tt = $K->ndarray($t);
        $tx = $K->ndarray($x);
        #$this->assertLessThan(0.00001,abs($mo->sum($dx)));
        $this->assertEquals($copyx->toArray(),$tx->toArray());
        $this->assertEquals($copyt->toArray(),$tt->toArray());

        $accuracy = $func->accuracy($t,$x);

        $x = $mo->array([
            [-10.0, -10.0 , 10.0],
            [ 10.0, -10.0 ,-10.0],
        ]);
        $t = $mo->array([1, 1]);
        $t = $K->array($t);
        $x = $K->array($x);
        //$y = $func->forward($x,true);
        $loss = $func->forward($t,$x);
        $this->assertGreaterThan(10.0,abs($loss));

        $dx = $func->backward([$K->array(1.0)]);
        $dx = $dx[0];
        $dx = $K->ndarray($dx);
        $this->assertLessThan(0.00001,abs($mo->sum($dx)));

        $x = $mo->array([
            [-2.0,  0.0, 2.0],
            [ 2.0, -2.0, 0.0],
        ]);
        $t = $mo->array([2, 2]);
        $this->assertTrue(
            $this->verifyGradient($mo,$K,$func,$t,$x,true));
    }
}
