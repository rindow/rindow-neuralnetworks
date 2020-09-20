<?php
namespace RindowTest\NeuralNetworks\Loss\SparseCategoricalCrossEntropyTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Loss\SparseCategoricalCrossEntropy;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Plot\Plot;

class Test extends TestCase
{
    public function verifyGradient($mo, $function, NDArray $t, NDArray $x,$fromLogits=null)
    {
        $f = function($x) use ($mo,$function,$t,$fromLogits){
            if($fromLogits) {
                $x = $function->forward($x,true);
            }
            $l = $function->loss($t,$x);
            return $mo->array([$l]);
        };
        $grads = $mo->la()->numericalGradient(1e-3,$f,$x);
        $outputs = $function->loss($t,$x);
        $dInputs = $function->differentiateLoss();
#echo "\n";
#echo "grads=".$mo->toString($grads[0],'%5.3f',true)."\n\n";
#echo "dInputs=".$mo->toString($dInputs,'%5.3f',true)."\n\n";
#echo $mo->asum($mo->op($grads[0],'-',$dInputs))."\n";
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
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $plt = new Plot($this->getPlotConfig(),$mo);
        $loss = $nn->losses()->SparseCategoricalCrossEntropy();
        $x = [[0.1,0.9],[0.3,0.7],[0.5,0.5],[0.7,0.3],[0.9,0.1]];
        $t = [0,0,0,0,0];
        $y = [];
        foreach($x as $k => $xx) {
            $tt = $t[$k];
            $y[] = $loss->loss($mo->array([$tt]),$mo->array([$xx]));
        }
        #$plt->plot($mo->array($y));
        $plt->show();
        $this->assertTrue(true);
    }

    public function testBuilder()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $this->assertInstanceof(
            'Rindow\NeuralNetworks\Loss\SparseCategoricalCrossEntropy',
            $nn->losses()->SparseCategoricalCrossEntropy());
    }

    public function testDefault()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $func = new SparseCategoricalCrossEntropy($backend);

        $x = $mo->array([
            [0.00001, 0.00001 , 0.99998],
            [0.99998, 0.00001 , 0.00001],
        ]);
        $t = $mo->array([2, 0],NDArray::int64);
        $copyx = $mo->copy($x);
        $copyt = $mo->copy($t);
        $loss = $func->loss($t,$x);
        $this->assertLessThan(0.001,abs($loss));
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());
        $dx = $func->differentiateLoss();
        //$this->assertLessThan(0.00001,abs($mo->sum($dx)));
        //$this->assertLessThan(0.001,abs(1-$dx[0][0])/6);
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        $accuracy = $func->accuracy($t,$x);

        $x = $mo->array([
            [0.00001, 0.00001 , 0.99998],
            [0.99998, 0.00001 , 0.00001],
        ]);
        $t = $mo->array([1, 1]);
        $loss = $func->loss($t,$x);
        $this->assertGreaterThan(10,abs($loss));
        $dx = $func->differentiateLoss();

        $x = $mo->array([
            [0.00001, 0.20000 , 0.79998],
            [0.79998, 0.00001 , 0.20000],
        ]);
        $t = $mo->array([2, 2]);
        $this->assertTrue(
            $this->verifyGradient($mo,$func,$t,$x));
    }

    public function testFromLogits()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $func = new SparseCategoricalCrossEntropy($backend);
        $func->setFromLogits(true);

        $x = $mo->array([
            [-10.0, -10.0 , 10.0],
            [ 10.0, -10.0 ,-10.0],
        ]);
        $t = $mo->array([2, 0],NDArray::int64);
        $y = $func->forward($x,true);
        $copyx = $mo->copy($x);
        $copyt = $mo->copy($t);
        $loss = $func->loss($t,$y);
        $this->assertLessThan(0.00001,abs($loss));
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        $dx = $func->differentiateLoss();
        #$this->assertLessThan(0.00001,abs($mo->sum($dx)));
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        $accuracy = $func->accuracy($t,$x);

        $x = $mo->array([
            [-10.0, -10.0 , 10.0],
            [ 10.0, -10.0 ,-10.0],
        ]);
        $t = $mo->array([1, 1]);
        $y = $func->forward($x,true);
        $loss = $func->loss($t,$y);
        $this->assertGreaterThan(10.0,abs($loss));

        $dx = $func->differentiateLoss();
        $this->assertLessThan(0.00001,abs($mo->sum($dx)));

        $x = $mo->array([
            [-2.0,  0.0, 2.0],
            [ 2.0, -2.0, 0.0],
        ]);
        $t = $mo->array([2, 2]);
        $this->assertTrue(
            $this->verifyGradient($mo,$func,$t,$x,true));
    }
}
