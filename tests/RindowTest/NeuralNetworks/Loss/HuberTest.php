<?php
namespace RindowTest\NeuralNetworks\Loss\HuberTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Loss\Huber;
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
            //if($fromLogits) {
            //    #$x = $function->forward($x,true);
            //    $x = $K->sigmoid($x);
            //}
            $l = $function->forward($t,$x);
            return $mo->array([$l]);
        };
        $xx = $K->ndarray($x);
        $grads = $mo->la()->numericalGradient(1e-3,$f,$xx);
        $outputs = $function->forward($t,$x);
        $dInputs = $function->backward([$K->array(1.0)]);
        $dInputs = $K->ndarray($dInputs[0]);
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
        $loss = $nn->losses()->Huber();
        $x = $mo->arange(400,-2.0,0.01);
        $t = $mo->zeros([400]);
        $y = [];
        foreach($x as $k => $xx) {
            $tt = $t[$k];
            $y[] = $loss->forward($K->array([[$tt]]),$K->array([[$xx]]));
        }
        $plt->plot($x,$mo->array($y));
        $plt->show();
        $this->assertTrue(true);
    }

    public function testBuilder()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $this->assertInstanceof(
            'Rindow\NeuralNetworks\Loss\Huber',
            $nn->losses()->Huber());
    }

    public function testDefault()
    {
        $mo = new MatrixOperator();
        $K = $backend = $this->newBackend($mo);
        $func = new Huber($backend);
        $func2 = new Huber($backend,['delta'=>2.0]);

        //
        //  Squared Loss with delta=1.0
        //
        $x = $K->array([
            [-0.5, -0.25, 0.5, 0.25],
        ]);
        $t = $K->array([
            [0.25, 0.25, 0.25, 0.25],
        ]);
        $copyx = $K->copy($x);
        $copyt = $K->copy($t);
        $loss = $func->forward($t,$x);
        #$accuracy = $func->accuracy($t,$x);
        $this->assertLessThan(1e-5, abs(0.109375-$loss));
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        [$dx] = $func->backward([$K->array(1.0)]);
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());
        $this->assertLessThan(1e-5, abs(-0.1875-$K->scalar($dx[0][0])));
        $this->assertLessThan(1e-5, abs(-0.125-$K->scalar($dx[0][1])));
        $this->assertLessThan(1e-5, abs(0.0625-$K->scalar($dx[0][2])));
        $this->assertLessThan(1e-5, abs(0-$K->scalar($dx[0][3])));

        $accuracy = $func->accuracy($t,$x);
        //$this->assertLessThan(0.0001,abs(1-$accuracy));
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        //
        //  Squared Loss with delta=2.0
        //
        $x = $K->array([
            [0.25, 0.5, 1.0, 1.5],
        ]);
        $t = $K->array([
            [0.25, 0.25, 0.25, 0.25],
        ]);
        $copyx = $K->copy($x);
        $copyt = $K->copy($t);
        $loss = $func2->forward($t,$x);
        #$accuracy = $func->accuracy($t,$x);
        $this->assertLessThan(1e-5, abs(0.2734375-$loss));
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        [$dx] = $func2->backward([$K->array(1.0)]);
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());
        $this->assertLessThan(1e-5, abs(0.0-$K->scalar($dx[0][0])));
        $this->assertLessThan(1e-5, abs(0.0625-$K->scalar($dx[0][1])));
        $this->assertLessThan(1e-5, abs(0.1875-$K->scalar($dx[0][2])));
        $this->assertLessThan(1e-5, abs(0.3125-$K->scalar($dx[0][3])));

        $accuracy = $func->accuracy($t,$x);
        //$this->assertLessThan(0.0001,abs(1-$accuracy));
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        //
        //  Linear Loss with delta=1.0
        //
        $x = $K->array([
            [-4.0, -2.0, 4.0, 2.0],
        ]);
        $t = $K->array([
            [1.0, 1.0, 1.0, 1.0],
        ]);
        $copyx = $K->copy($x);
        $copyt = $K->copy($t);
        $loss = $func->forward($t,$x);
        #$accuracy = $func->accuracy($t,$x);
        $this->assertLessThan(1e-5, abs(2.5-$loss));
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        [$dx] = $func->backward([$K->array(1.0)]);
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());
        $this->assertLessThan(1e-5, abs(-0.25-$K->scalar($dx[0][0])));
        $this->assertLessThan(1e-5, abs(-0.25-$K->scalar($dx[0][1])));
        $this->assertLessThan(1e-5, abs(0.25-$K->scalar($dx[0][2])));
        $this->assertLessThan(1e-5, abs(0.25-$K->scalar($dx[0][3])));

        $accuracy = $func->accuracy($t,$x);
        //$this->assertLessThan(0.0001,abs(1-$accuracy));
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        //
        //  Linear Loss with delta=2.0
        //
        $x = $K->array([
            [-4.0, -2.0, 4.0, 2.0],
        ]);
        $t = $K->array([
            [1.0, 1.0, 1.0, 0.0],
        ]);
        $copyx = $K->copy($x);
        $copyt = $K->copy($t);
        $loss = $func2->forward($t,$x);
        #$accuracy = $func->accuracy($t,$x);
        $this->assertLessThan(1e-5, abs(4.5-$loss));
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        [$dx] = $func2->backward([$K->array(1.0)]);
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());
        $this->assertLessThan(1e-5, abs(-0.5-$K->scalar($dx[0][0])));
        $this->assertLessThan(1e-5, abs(-0.5-$K->scalar($dx[0][1])));
        $this->assertLessThan(1e-5, abs( 0.5-$K->scalar($dx[0][2])));
        $this->assertLessThan(1e-5, abs( 0.5-$K->scalar($dx[0][3])));

        $accuracy = $func->accuracy($t,$x);
        //$this->assertLessThan(0.0001,abs(1-$accuracy));
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        //
        // verifyGradient
        //

        $x = $K->array([
            [0.001,],
            [0.999,],
            [1.1,],
            [1.5,],
        ]);
        $t = $K->array([
            [0,],
            [0,],
            [0,],
            [0,],
        ]);
        $this->assertTrue(
            $this->verifyGradient($mo,$K,$func,$t,$x));
    }

    public function testImmediatelyValue()
    {
        $mo = new MatrixOperator();
        $K = $backend = $this->newBackend($mo);
        $func = new Huber($backend);
        $func2 = new Huber($backend,['delta'=>2.0]);

        $x = $K->array([
            0.001,
            0.999,
            1.1,
            1.5,
        ]);
        $t = $K->array([
            0,
            0,
            0,
            0,
        ]);

        $loss = $func->forward($t,$x);
        [$dx] = $func->backward([$K->array(1.0)]);
        $this->assertLessThan(1e-5, abs(0.00025-$K->scalar($dx[0])));
        $this->assertLessThan(1e-5, abs(0.24975-$K->scalar($dx[1])));
        $this->assertLessThan(1e-5, abs(0.25-$K->scalar($dx[2])));
        $this->assertLessThan(1e-5, abs(0.25-$K->scalar($dx[3])));
    }
}
