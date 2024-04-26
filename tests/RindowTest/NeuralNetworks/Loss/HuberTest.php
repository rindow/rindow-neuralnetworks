<?php
namespace RindowTest\NeuralNetworks\Loss\HuberTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Loss\Huber;
use Rindow\NeuralNetworks\Metric\MetricCatalog;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Plot\Plot;

class HuberTest extends TestCase
{
    public function newMatrixOperator()
    {
        return new MatrixOperator();
    }

    public function newNeuralNetworks($mo)
    {
        return new NeuralNetworks($mo);
    }

    protected function accuracy($nn,$loss,$trues,$predicts) : float
    {
        $metric = $loss->accuracyMetric();
        $metricObject = MetricCatalog::factory($nn->backend(),$metric);
        $metricObject->update($trues,$predicts);
        $accuracy = $metricObject->result();
        return $accuracy;
    }

    public function verifyGradient($mo, $nn, $K, $g, $function, NDArray $t, NDArray $x,$fromLogits=null)
    {
        $f = function($x) use ($mo,$K,$g,$function,$t,$fromLogits){
            $x = $K->array($x);
            //if($fromLogits) {
            //    #$x = $function->forward($x,true);
            //    $x = $K->sigmoid($x);
            //}
            $l = $function->forward($t,$x);
            $y = $g->mul($l,$g->Variable(5));
            return $K->ndarray($y);
        };
        $xx = $K->ndarray($x);
        $grads = $mo->la()->numericalGradient(1e-3,$f,$xx);
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($function,$t, $x, $g) {
                $l = $function->forward($t, $x);
                $y = $g->mul($l,$g->Variable(5));
                return $y;
            }
        );
        $mul = $y->creator();
        $outVar = $mul->inputs()[0];
        $outputs = $K->scalar($y);
        $loss = $outVar->creator();
//echo $loss->name()."\n";
        $dInputs = $mul->backward([$K->onesLike($y)]);
//echo "mul dInput[0]=".$mo->toString($dInputs[0],'%5.5f',true)."\n\n";
        $dOutputs = [$dInputs[0]];
        $dInputs = $loss->backward($dOutputs);
//echo "\n";
//echo "grads=".$mo->toString($grads[0],'%5.5f',true)."\n\n";
//echo "dInputs=".$mo->toString($dInputs[1],'%5.5f',true)."\n\n";
        $dInputs = $K->ndarray($dInputs[1]);
//echo $mo->asum($mo->op($grads[0],'-',$dInputs))."\n";
        return $mo->la()->isclose($grads[0],$dInputs,null,1e-4);
    }

    public function getPlotConfig()
    {
        return [
            'renderer.skipCleaning' => true,
            'renderer.skipRunViewer' => getenv('PLOT_RENDERER_SKIP') ? true : false,
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
            Huber::class,
            $nn->losses()->Huber());
    }

    public function testDefault()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $func = $nn->losses()->Huber();
        $func2 = $nn->losses()->Huber(delta:2.0);

        //
        //  Squared Loss with delta=1.0
        //
        $x = $K->array([
            [-0.5, -0.25, 0.5, 0.25],
            [-0.5, -0.25, 0.5, 0.25],
        ]);
        $t = $K->array([
            [0.25, 0.25, 0.25, 0.25],
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
        #$accuracy = $this->accuracy($nn,$func,$t,$x);
        $this->assertLessThan(1e-5, abs(0.109375-$loss));
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        [$dmy,$dx] = $outputsVariable->creator()->backward([$K->array(1.0)]);
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());
        $dx = $K->ndarray($dx);
        $this->assertTrue($mo->la()->isclose($mo->la()->array([
                [-0.093750,-0.062500, 0.031250, 0.000000],
                [-0.093750,-0.062500, 0.031250, 0.000000],
        ]),$dx));

        $accuracy = $this->accuracy($nn,$func,$t,$x);
        $accuracy = $K->scalar($accuracy);
        //$this->assertLessThan(0.0001,abs(1-$accuracy));
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        //
        //  Squared Loss with delta=2.0
        //
        $x = $K->array([
            [0.25, 0.5, 1.0, 1.5],
            [0.25, 0.5, 1.0, 1.5],
        ]);
        $t = $K->array([
            [0.25, 0.25, 0.25, 0.25],
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
        #$accuracy = $this->accuracy($nn,$func,$t,$x);
        $this->assertLessThan(1e-5, abs(0.2734375-$loss));
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        [$dmy,$dx] = $outputsVariable->creator()->backward([$K->array(1.0)]);
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());
        $dx = $K->ndarray($dx);
        $this->assertTrue($mo->la()->isclose($mo->la()->array([
            [0.0, 0.03125, 0.09375, 0.15625],
            [0.0, 0.03125, 0.09375, 0.15625],
        ]),$dx));

        $accuracy = $this->accuracy($nn,$func,$t,$x);
        $accuracy = $K->scalar($accuracy);
        //$this->assertLessThan(0.0001,abs(1-$accuracy));
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        //
        //  Linear Loss with delta=1.0
        //
        $x = $K->array([
            [-4.0, -2.0, 4.0, 2.0],
            [-4.0, -2.0, 4.0, 2.0],
        ]);
        $t = $K->array([
            [1.0, 1.0, 1.0, 1.0],
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
        #$accuracy = $this->accuracy($nn,$func,$t,$x);
        $this->assertLessThan(1e-5, abs(2.5-$loss));
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        [$dmy,$dx] = $outputsVariable->creator()->backward([$K->array(1.0)]);
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        $dx = $K->ndarray($dx);
        $this->assertTrue($mo->la()->isclose($mo->la()->array([
            [-0.125, -0.125,  0.125,  0.125],
            [-0.125, -0.125,  0.125,  0.125]
        ]),$dx));

        $accuracy = $this->accuracy($nn,$func,$t,$x);
        //$this->assertLessThan(0.0001,abs(1-$accuracy));
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        //
        //  Linear Loss with delta=2.0
        //
        $x = $K->array([
            [-4.0, -2.0, 4.0, 2.0],
            [-4.0, -2.0, 4.0, 2.0],
        ]);
        $t = $K->array([
            [1.0, 1.0, 1.0, 0.0],
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
        
        #$accuracy = $this->accuracy($nn,$func,$t,$x);
        $this->assertLessThan(1e-5, abs(4.5-$loss));
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        [$dmy,$dx] = $outputsVariable->creator()->backward([$K->array(1.0)]);
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());
        $dx = $K->ndarray($dx);
        $this->assertTrue($mo->la()->isclose($mo->la()->array([
            [-0.25, -0.25,  0.25,  0.25],
            [-0.25, -0.25,  0.25,  0.25]
        ]),$dx));

        $accuracy = $this->accuracy($nn,$func,$t,$x);
        //$this->assertLessThan(0.0001,abs(1-$accuracy));
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        //
        // verifyGradient
        //

        $x = $K->array([
            [0.001,0.001],
            [0.999,0.999],
            [1.1,1.1],
            [1.5,1.5],
        ]);
        $t = $K->array([
            [0,0],
            [0,0],
            [0,0],
            [0,0],
        ]);
        $this->assertTrue(
            $this->verifyGradient($mo,$nn,$K,$g,$func,$t,$x));
    }

    public function testReductionNone()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $func = $nn->losses()->Huber(reduction:'none');
        $func2 = $nn->losses()->Huber(delta:2.0,reduction:'none');

        //
        //  Squared Loss with delta=1.0
        //
        $x = $K->array([
            [-0.5, -0.25, 0.5, 0.25],
            [-0.5, -0.25, 0.5, 0.25],
        ]);
        $t = $K->array([
            [0.25, 0.25, 0.25, 0.25],
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
        $loss = $K->ndarray($outputsVariable);
        #$accuracy = $this->accuracy($nn,$func,$t,$x);
        $this->assertTrue($mo->la()->isclose($mo->la()->array(
            [0.109375, 0.109375],
        ),$loss));
        //$this->assertLessThan(1e-5, abs(0.109375-$loss));
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        [$dmy,$dx] = $outputsVariable->creator()->backward([$K->array([1.0,1.0])]);
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());
        $dx = $K->ndarray($dx);
        $this->assertTrue($mo->la()->isclose($mo->la()->array([
                [-0.1875, -0.125 ,  0.0625,  0.0    ],
                [-0.1875, -0.125 ,  0.0625,  0.0    ],
        ]),$dx));

        $accuracy = $this->accuracy($nn,$func,$t,$x);
        $accuracy = $K->scalar($accuracy);
        //$this->assertLessThan(0.0001,abs(1-$accuracy));
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        //
        // verifyGradient
        //

        $x = $K->array([
            [0.001,0.001],
            [0.999,0.999],
            [1.1,1.1],
            [1.5,1.5],
        ]);
        $t = $K->array([
            [0,0],
            [0,0],
            [0,0],
            [0,0],
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
        $func = $nn->losses()->Huber();
        $func2 = $nn->losses()->Huber(delta:2.0);

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
