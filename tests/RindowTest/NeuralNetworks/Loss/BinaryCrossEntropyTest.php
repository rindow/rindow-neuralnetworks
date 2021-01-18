<?php
namespace RindowTest\NeuralNetworks\Loss\BinaryCrossEntropyTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Loss\BinaryCrossEntropy;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Plot\Plot;

class Test extends TestCase
{
    public function newBackend($mo)
    {
        $builder = new NeuralNetworks($mo);
        return $builder->backend();
    }

    public function verifyGradient($mo, $K, $function, NDArray $t, NDArray $x,$fromLogits=null)
    {
        $f = function($x) use ($mo,$K,$function,$t,$fromLogits){
            $x = $K->array($x);
            if($fromLogits) {
                $x = $function->forward($x,true);
            }
            $l = $function->loss($t,$x);
            return $mo->array([$l]);
        };
        $xx = $K->ndarray($x);
        $grads = $mo->la()->numericalGradient(1e-3,$f,$xx);
        $outputs = $function->loss($t,$x);
        $dInputs = $function->differentiateLoss();
        $dInputs = $K->ndarray($dInputs);
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
        $K = $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $plt = new Plot($this->getPlotConfig(),$mo);
        $loss = $nn->losses()->BinaryCrossEntropy();
        $x = [0.1,0.3,0.5,0.7,0.9,0.1,0.3,0.5,0.7,0.9];
        $t = [1,1,1,1,1,0,0,0,0,0];
        $y = [];
        foreach($x as $k => $xx) {
            $tt = $t[$k];
            $y[] = $loss->loss($K->array([$tt]),$K->array([[$xx]]));
        }
        $plt->plot($mo->array($y));
        #$plt->show();
        $this->assertTrue(true);
    }

    public function testBuilder()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $this->assertInstanceof(
            'Rindow\NeuralNetworks\Loss\BinaryCrossEntropy',
            $nn->losses()->BinaryCrossEntropy());
    }

    public function testDefault()
    {
        $mo = new MatrixOperator();
        $K = $backend = $this->newBackend($mo);
        $func = new BinaryCrossEntropy($backend);

        $x = $K->array([
            [0.00001], [0.00001] , [0.99999],
        ]);
        $t = $K->array([
            0.0, 0.0 , 1.0,
        ]);
        $copyx = $K->copy($x);
        $copyt = $K->copy($t);
        $loss = $func->loss($t,$x);
        #$accuracy = $func->accuracy($t,$x);
        $this->assertLessThan(0.001,abs($loss));
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        $dx = $func->differentiateLoss();
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());
        //$this->assertLessThan(0.001,abs(1-$dx[0][0])/6);
        //$this->assertLessThan(0.001,abs(1-$dx[1][0])/6);
        //$this->assertLessThan(0.001,abs(-1-$dx[2][0])/6);

        $accuracy = $func->accuracy($t,$x);
        $this->assertLessThan(0.0001,abs(1-$accuracy));
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());


        $x = $K->array([
            [0.9999],[0.9999],[0.0001],
        ]);
        $t = $K->array([
            0.0, 0.0 , 1.0,
        ]);
        $loss = $func->loss($t,$x);
        $this->assertGreaterThan(8,abs($loss));

        $dx = $func->differentiateLoss();
        $dx = $K->ndarray($dx);
        $this->assertGreaterThan(100,$dx[0][0]);
        $this->assertGreaterThan(100,$dx[1][0]);
        $this->assertLessThan(100,$dx[2][0]);

        $accuracy = $func->accuracy($t,$x);
        $this->assertLessThan(0.0001,abs(0-$accuracy));

        $x = $K->array([
            [0.001,],
            [0.999,],
            [0.5,],
        ]);
        $t = $K->array([
            0, 1 , 1,
        ]);
        $this->assertTrue(
            $this->verifyGradient($mo,$K,$func,$t,$x));
    }

    public function testFromLogits()
    {
        $mo = new MatrixOperator();
        $K = $backend = $this->newBackend($mo);
        $func = new BinaryCrossEntropy($backend);
        $func->setFromLogits(true);

        $x = $K->array([
            [-10.0], [-10.0] , [10.0],
        ]);
        $t = $K->array([
            0.0, 0.0 , 1.0,
        ]);
        $copyx = $K->copy($x);
        $copyt = $K->copy($t);
        $y = $func->forward($x,true);
        $loss = $func->loss($t,$y);
        $this->assertLessThan(0.001,abs($loss));
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        $dx = $func->differentiateLoss();
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        $x = $K->array([
            [10.0], [-10.0] , [10.0],
        ]);
        $t = $K->array([
            0.0, 1.0 , 0.0,
        ]);
        $copyx = $K->copy($x);
        $copyt = $K->copy($t);
        $y = $func->forward($x,true);
        $loss = $func->loss($t,$y);
        $this->assertGreaterThan(7,abs($loss));
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        $dx = $func->differentiateLoss();
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        $x = $K->array([
            [-2.0], [2.0] , [0.0],
        ]);
        $t = $K->array([
            0.0, 1.0 , 0.0,
        ]);
        $this->assertTrue(
            $this->verifyGradient($mo,$K,$func,$t,$x,true));
    }
}
