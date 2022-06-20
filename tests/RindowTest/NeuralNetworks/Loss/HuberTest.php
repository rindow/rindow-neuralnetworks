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
    public function newMatrixOperator()
    {
        return new MatrixOperator();
    }

    public function newNeuralNetworks($mo)
    {
        return new NeuralNetworks($mo);
    }

    public function verifyGradient($mo, $nn, $K, $g, $function, NDArray $t, NDArray $x,$fromLogits=null)
    {
        $f = function($x) use ($mo,$K,$function,$t,$fromLogits){
            $x = $K->array($x);
            //if($fromLogits) {
            //    #$x = $function->forward($x,true);
            //    $x = $K->sigmoid($x);
            //}
            $l = $function->forward($t,$x);
            return $mo->array([$K->scalar($l)]);
        };
        $xx = $K->ndarray($x);
        $grads = $mo->la()->numericalGradient(1e-3,$f,$xx);
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($function,$t, $x) {
                $outputsVariable = $function->forward($t, $x);
                return $outputsVariable;
            }
        );
        $dInputs = $outputsVariable->creator()->backward([$K->array(1.0)]);
        $dInputs = $K->ndarray($dInputs[1]);
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
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);
        $loss = $nn->losses()->Huber();
        $x = $mo->arange(400,-2.0,0.01);
        $t = $mo->zeros([400]);
        $y = [];
        foreach($x as $k => $xx) {
            $tt = $t[$k];
            $y[] = $K->scalar($loss->forward($K->array([[$tt]]),$K->array([[$xx]])));
        }
        $plt->plot($x,$mo->array($y));
        $plt->show();
        $this->assertTrue(true);
    }

    public function testBuilder()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $this->assertInstanceof(
            'Rindow\NeuralNetworks\Loss\Huber',
            $nn->losses()->Huber());
    }

    public function testDefault()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $func = new Huber($K);
        $func2 = new Huber($K, delta:2.0);

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
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($func,$t, $x) {
                $outputsVariable = $func->forward($t, $x);
                return $outputsVariable;
            }
        );
        $loss = $K->scalar($outputsVariable);
        #$accuracy = $func->accuracy($t,$x);
        $this->assertLessThan(1e-5, abs(0.109375-$loss));
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        [$dmy,$dx] = $outputsVariable->creator()->backward([$K->array(1.0)]);
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());
        $this->assertLessThan(1e-5, abs(-0.1875-$K->scalar($dx[0][0])));
        $this->assertLessThan(1e-5, abs(-0.125-$K->scalar($dx[0][1])));
        $this->assertLessThan(1e-5, abs(0.0625-$K->scalar($dx[0][2])));
        $this->assertLessThan(1e-5, abs(0-$K->scalar($dx[0][3])));

        $accuracy = $func->accuracy($t,$x);
        $accuracy = $K->scalar($accuracy);
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
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($func2,$t, $x) {
                $outputsVariable = $func2->forward($t, $x);
                return $outputsVariable;
            }
        );
        $loss = $K->scalar($outputsVariable);
        #$accuracy = $func->accuracy($t,$x);
        $this->assertLessThan(1e-5, abs(0.2734375-$loss));
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        [$dmy,$dx] = $outputsVariable->creator()->backward([$K->array(1.0)]);
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());
        $this->assertLessThan(1e-5, abs(0.0-$K->scalar($dx[0][0])));
        $this->assertLessThan(1e-5, abs(0.0625-$K->scalar($dx[0][1])));
        $this->assertLessThan(1e-5, abs(0.1875-$K->scalar($dx[0][2])));
        $this->assertLessThan(1e-5, abs(0.3125-$K->scalar($dx[0][3])));

        $accuracy = $func->accuracy($t,$x);
        $accuracy = $K->scalar($accuracy);
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
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($func,$t, $x) {
                $outputsVariable = $func->forward($t, $x);
                return $outputsVariable;
            }
        );
        $loss = $K->scalar($outputsVariable);
        #$accuracy = $func->accuracy($t,$x);
        $this->assertLessThan(1e-5, abs(2.5-$loss));
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        [$dmy,$dx] = $outputsVariable->creator()->backward([$K->array(1.0)]);
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
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($func2,$t, $x) {
                $outputsVariable = $func2->forward($t, $x);
                return $outputsVariable;
            }
        );
        $loss = $K->scalar($outputsVariable);
        
        #$accuracy = $func->accuracy($t,$x);
        $this->assertLessThan(1e-5, abs(4.5-$loss));
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        [$dmy,$dx] = $outputsVariable->creator()->backward([$K->array(1.0)]);
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
            $this->verifyGradient($mo,$nn,$K,$g,$func,$t,$x));
    }

    public function testImmediatelyValue()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $func = new Huber($K);
        $func2 = new Huber($K, delta:2.0);

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

        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($func,$t, $x) {
                $outputsVariable = $func->forward($t, $x);
                return $outputsVariable;
            }
        );
        $loss = $K->scalar($outputsVariable);
        [$dmy,$dx] = $outputsVariable->creator()->backward([$K->array(1.0)]);
        $this->assertLessThan(1e-5, abs(0.00025-$K->scalar($dx[0])));
        $this->assertLessThan(1e-5, abs(0.24975-$K->scalar($dx[1])));
        $this->assertLessThan(1e-5, abs(0.25-$K->scalar($dx[2])));
        $this->assertLessThan(1e-5, abs(0.25-$K->scalar($dx[3])));
    }
}
