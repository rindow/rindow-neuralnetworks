<?php
namespace RindowTest\NeuralNetworks\Loss\BinaryCrossEntropyTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Loss\BinaryCrossEntropy;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Plot\Plot;

class BinaryCrossEntropyTest extends TestCase
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
        $f = function($x) use ($mo,$K,$function,$t,$fromLogits, $g){
            $x = $K->array($x);
            $t = $K->array($t);
            //if($fromLogits) {
            //    $x = $function->forward($x,true);
            //}
            $l = $function->forward($t,$x);
            $y = $g->mul($l,$g->Variable(5));
            return $K->ndarray($y);
        };
        $xx = $K->ndarray($x);
        $grads = $mo->la()->numericalGradient(1e-3,$f,$xx);
        $x = $K->array($x);
        $t = $K->array($t);
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
/*
    public function verifyGradient($mo, $nn, $K, $g, $function, NDArray $t, NDArray $x,$fromLogits=null)
    {
        $f = function($x) use ($mo,$K,$function,$t,$fromLogits){
            $x = $K->array($x);
            //if($fromLogits) {
            //    #$x = $function->forward($x,true);
            //    $x = $K->sigmoid($x);
            //}
            $l = $function->forward($t,$x);
            $l = $K->scalar($l);
            return $mo->array([$l]);
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
*/
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
        $loss = $nn->losses()->BinaryCrossEntropy();
        $x = [0.1,0.3,0.5,0.7,0.9,0.1,0.3,0.5,0.7,0.9];
        $t = [1,1,1,1,1,0,0,0,0,0];
        $y = [];
        foreach($x as $k => $xx) {
            $tt = $t[$k];
            $y[] = $K->scalar($loss->forward($K->array([$tt]),$K->array([[$xx]])));
        }
        $plt->plot($mo->array($y));
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
            BinaryCrossEntropy::class,
            $nn->losses()->BinaryCrossEntropy());
    }

    public function testDefault()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $func = $nn->losses()->BinaryCrossEntropy();

        $x = $K->array([
            [0.00001], [0.00001] , [0.99999],
        ]);
        $t = $K->array([
            [0.0], [0.0] , [1.0],
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
        $this->assertLessThan(0.001,abs($loss));
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        $dx = $outputsVariable->creator()->backward([$K->array(1.0)]);
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());
        //$this->assertLessThan(0.001,abs(1-$dx[0][0])/6);
        //$this->assertLessThan(0.001,abs(1-$dx[1][0])/6);
        //$this->assertLessThan(0.001,abs(-1-$dx[2][0])/6);

        $accuracy = $func->accuracy($t,$x);
        $accuracy = $K->scalar($accuracy);
        $this->assertLessThan(0.0001,abs(1-$accuracy));
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());


        $x = $K->array([
            [0.9999],[0.9999],[0.0001],
        ]);
        $t = $K->array([
            [0.0], [0.0] , [1.0],
        ]);
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($func,$t, $x) {
                $outputsVariable = $func->forward($t, $x);
                return $outputsVariable;
            }
        );
        $loss = $K->scalar($outputsVariable);
        $this->assertGreaterThan(8,abs($loss));

        $dx = $outputsVariable->creator()->backward([$K->array(1.0)]);
        $dx = $K->ndarray($dx[1]);

        $this->assertGreaterThan(100,$dx[0][0]);
        $this->assertGreaterThan(100,$dx[1][0]);
        $this->assertLessThan(100,$dx[2][0]);

        $accuracy = $func->accuracy($t,$x);
        $accuracy = $K->scalar($accuracy);
        $this->assertLessThan(0.0001,abs(0-$accuracy));

        $x = $K->array([
            [0.001,],
            [0.999,],
            [0.5,],
        ]);
        $t = $K->array([
            [0], [1] , [1],
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
        $func = $nn->losses()->BinaryCrossEntropy(reduction:'none');

        $x = $K->array([
            [0.00001], [0.00001] , [0.99999],
        ]);
        $t = $K->array([
            [0.0], [0.0] , [1.0],
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
        #$accuracy = $func->accuracy($t,$x);
        //$this->assertLessThan(0.001,abs($loss));
        $this->assertTrue($mo->la()->isclose(
            $loss,$mo->la()->array([1e-5, 1e-5, 1e-5]))
        );
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        $dx = $outputsVariable->creator()->backward([$K->array([1.0,1.0,1.0])]);
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());
        //$this->assertLessThan(0.001,abs(1-$dx[0][0])/6);
        //$this->assertLessThan(0.001,abs(1-$dx[1][0])/6);
        //$this->assertLessThan(0.001,abs(-1-$dx[2][0])/6);

        $accuracy = $func->accuracy($t,$x);
        $accuracy = $K->scalar($accuracy);
        $this->assertLessThan(0.0001,abs(1-$accuracy));
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());


        $x = $K->array([
            [0.9999],[0.9999],[0.0001],
        ]);
        $t = $K->array([
            [0.0], [0.0] , [1.0],
        ]);
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($func,$t, $x) {
                $outputsVariable = $func->forward($t, $x);
                return $outputsVariable;
            }
        );
        $loss = $K->ndarray($outputsVariable);
        //$this->assertGreaterThan(8,abs($loss));
        $this->assertTrue($mo->la()->isclose(
            $loss,$mo->la()->array([9.21012, 9.2102, 9.2103]))
        );

        $dx = $outputsVariable->creator()->backward([$K->array([1.0,1.0,1.0])]);
        $dx = $K->ndarray($dx[1]);

        $this->assertGreaterThan(100,$dx[0][0]);
        $this->assertGreaterThan(100,$dx[1][0]);
        $this->assertLessThan(100,$dx[2][0]);

        $accuracy = $func->accuracy($t,$x);
        $accuracy = $K->scalar($accuracy);
        $this->assertLessThan(0.0001,abs(0-$accuracy));

        $x = $K->array([
            [0.001,],
            [0.999,],
            [0.5,],
        ]);
        $t = $K->array([
            [0], [1] , [1],
        ]);
        $this->assertTrue(
            $this->verifyGradient($mo,$nn,$K,$g,$func,$t,$x));
    }

    public function testFromLogits()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $func = $nn->losses()->BinaryCrossEntropy(from_logits:true);

        $x = $K->array([
            [-10.0], [-10.0] , [10.0],
        ]);
        $t = $K->array([
            [0.0], [0.0] , [1.0],
        ]);
        $copyx = $K->copy($x);
        $copyt = $K->copy($t);
        //$y = $func->forward($x,true);
        //$y = $backend->sigmoid($x);
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($func,$t, $x) {
                $outputsVariable = $func->forward($t, $x);
                return $outputsVariable;
            }
        );
        $loss = $K->scalar($outputsVariable);
        $this->assertLessThan(0.001,abs($loss));
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        $dx = $outputsVariable->creator()->backward([$K->array(1.0)]);
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        $x = $K->array([
            [10.0], [-10.0] , [10.0],
        ]);
        $t = $K->array([
            [0.0], [1.0] , [0.0],
        ]);
        $copyx = $K->copy($x);
        $copyt = $K->copy($t);
        //$y = $func->forward($x,true);
        //$y = $backend->sigmoid($x);
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($func,$t, $x) {
                $outputsVariable = $func->forward($t, $x);
                return $outputsVariable;
            }
        );
        $loss = $K->scalar($outputsVariable);
        $this->assertGreaterThan(7,abs($loss));
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        $dx = $outputsVariable->creator()->backward([$K->array(1.0)]);
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        $x = $K->array([
            [-2.0], [2.0] , [0.0],
        ]);
        $t = $K->array([
            0.0, 1.0 , 0.0,
        ]);
        $this->assertTrue(
            $this->verifyGradient($mo,$nn,$K,$g,$func,$t,$x,true));
    }

    public function testNoReductionAndFromLogits()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $func = $nn->losses()->BinaryCrossEntropy(from_logits:true,reduction:'none');

        $x = $K->array([
            [-10.0], [-10.0] , [10.0],
        ]);
        $t = $K->array([
            [0.0], [0.0] , [1.0],
        ]);
        $copyx = $K->copy($x);
        $copyt = $K->copy($t);
        //$y = $func->forward($x,true);
        //$y = $backend->sigmoid($x);
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($func,$t, $x) {
                $outputsVariable = $func->forward($t, $x);
                return $outputsVariable;
            }
        );
        $loss = $K->ndarray($outputsVariable);
        //$this->assertLessThan(0.001,abs($loss));
        $this->assertTrue($mo->la()->isclose(
            $loss,$mo->la()->array([4.542e-5, 4.542e-5, 4.542e-5]))
        );
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        $dx = $outputsVariable->creator()->backward([$K->array([1.0,1.0,1.0])]);
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        $x = $K->array([
            [10.0], [-10.0] , [10.0],
        ]);
        $t = $K->array([
            [0.0], [1.0] , [0.0],
        ]);
        $copyx = $K->copy($x);
        $copyt = $K->copy($t);
        //$y = $func->forward($x,true);
        //$y = $backend->sigmoid($x);
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($func,$t, $x) {
                $outputsVariable = $func->forward($t, $x);
                return $outputsVariable;
            }
        );
        $loss = $K->ndarray($outputsVariable);
        //$this->assertGreaterThan(7,abs($loss));
        $this->assertTrue($mo->la()->isclose(
            $loss,$mo->la()->array([10, 10, 10]))
        );
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        $dx = $outputsVariable->creator()->backward([$K->array([1.0,1.0,1.0])]);
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        $x = $K->array([
            [-2.0], [2.0] , [0.0],
        ]);
        $t = $K->array([
            [0.0], [1.0] , [0.0],
        ]);
        $this->assertTrue(
            $this->verifyGradient($mo,$nn,$K,$g,$func,$t,$x,true));
    }

    public function testMultiBatchDimsLogitsFalseReductionSum()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();

        $func = $nn->losses()->BinaryCrossEntropy(from_logits:false,reduction:'sum');
        $x = $g->Variable($K->zeros([2,3,4]));
        $t = $g->Variable($K->zeros([2,3,4]));
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($func,$t, $x, $g) {
                $outputsVariable = $func->forward($t, $x);
                $y = $g->mul($outputsVariable,$g->Variable(2));
                return $y;
            }
        );
        $grads = $tape->gradient($y,$x);
        $this->assertEquals([],$y->shape());
        $this->assertEquals([2,3,4],$grads->shape());
    }

    public function testMultiBatchDimsLogitsTrueReductionSum()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();

        $func = $nn->losses()->BinaryCrossEntropy(from_logits:true,reduction:'sum');
        $x = $g->Variable($K->zeros([2,3,4]));
        $t = $g->Variable($K->zeros([2,3,4]));
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($func,$t, $x, $g) {
                $outputsVariable = $func->forward($t, $x);
                $y = $g->mul($outputsVariable,$g->Variable(2));
                return $y;
            }
        );
        $grads = $tape->gradient($y,$x);
        $this->assertEquals([],$y->shape());
        $this->assertEquals([2,3,4],$grads->shape());
    }

    public function testMultiBatchDimsLogitsFalseReductionNone()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();

        $func = $nn->losses()->BinaryCrossEntropy(from_logits:false,reduction:'none');
        $x = $g->Variable($K->zeros([2,3,4]));
        $t = $g->Variable($K->zeros([2,3,4]));
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($func,$t, $x, $g) {
                $outputsVariable = $func->forward($t, $x);
                $y = $g->mul($outputsVariable,$g->Variable(2));
                return $y;
            }
        );
        $grads = $tape->gradient($y,$x);
        $this->assertEquals([2,3],$y->shape());
        $this->assertEquals([2,3,4],$grads->shape());
    }

    public function testMultiBatchDimsLogitsTrueReductionNone()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();

        $func = $nn->losses()->BinaryCrossEntropy(from_logits:true,reduction:'none');
        $x = $g->Variable($K->zeros([2,3,4]));
        $t = $g->Variable($K->zeros([2,3,4]));
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($func,$t, $x, $g) {
                $outputsVariable = $func->forward($t, $x);
                $y = $g->mul($outputsVariable,$g->Variable(2));
                return $y;
            }
        );
        $grads = $tape->gradient($y,$x);
        $this->assertEquals([2,3],$y->shape());
        $this->assertEquals([2,3,4],$grads->shape());
    }
}
