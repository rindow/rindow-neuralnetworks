<?php
namespace RindowTest\NeuralNetworks\Loss\SparseCategoricalCrossEntropyTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Loss\SparseCategoricalCrossEntropy;
use Rindow\NeuralNetworks\Metric\MetricCatalog;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Plot\Plot;

class SparseCategoricalCrossEntropyTest extends TestCase
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
        $grads = $mo->la()->numericalGradient(1e-3,$f,$x);
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
        return $mo->la()->isclose($grads[0],$dInputs,null,1e-3);
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
        $loss = $nn->losses()->SparseCategoricalCrossEntropy();
        $x = [[0.1,0.9],[0.3,0.7],[0.5,0.5],[0.7,0.3],[0.9,0.1]];
        $t = [0,0,0,0,0];
        $y = [];
        foreach($x as $k => $xx) {
            $tt = $t[$k];
            $y[] = $K->scalar($loss->forward($K->array([$tt],dtype:NDArray::int32),$K->array([$xx])));
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
            SparseCategoricalCrossEntropy::class,
            $nn->losses()->SparseCategoricalCrossEntropy());
    }

    public function testDefault()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $func = $nn->losses()->SparseCategoricalCrossEntropy();

        //
        // match
        //
        
        $x = $mo->array([
            [0.00000, 0.00000 , 1.00000],
            [0.99998, 0.00001 , 0.00001],
        ]);
        $t = $mo->array([2, 0],dtype:NDArray::int32);
        $copyx = $mo->copy($x);
        $copyt = $mo->copy($t);
        $t = $K->array($t);
        $x = $K->array($x);
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($func,$t, $x, $g) {
                $outputsVariable = $func->forward($t, $x);
                $y = $g->mul($outputsVariable,$g->Variable(2));
                return $y;
            }
        );
        $loss = $K->scalar($y);
        $tt = $K->ndarray($t);
        $tx = $K->ndarray($x);
        $this->assertLessThan(1e-5,abs($loss-2.026e-05));
        $this->assertEquals($copyx->toArray(),$tx->toArray());
        $this->assertEquals($copyt->toArray(),$tt->toArray());
        $mul = $y->creator();
        $dx = $mul->backward([$K->array(1.0)]);
        $lossfunc = $mul->inputs()[0]->creator();
        $dx = $lossfunc->backward([$dx[0]]);
        $dx = $dx[1];
        $dx = $K->ndarray($dx);
        //$this->assertLessThan(0.00001,abs($mo->sum($dx)));
        //$this->assertLessThan(0.001,abs(1-$dx[0][0])/6);
        $this->assertEquals($copyx->toArray(),$tx->toArray());
        $this->assertEquals($copyt->toArray(),$tt->toArray());

        $accuracy = $this->accuracy($nn,$func,$t,$x);
        $accuracy = $K->scalar($accuracy);

        //
        // unmatch
        //

        $x = $mo->array([
            [0.00000, 0.00000 , 1.00000],
            [0.99998, 0.00001 , 0.00001],
        ]);
        $t = $mo->array([1, 1],dtype:NDArray::int32);
        $t = $K->array($t);
        $x = $K->array($x);
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($func,$t, $x, $g) {
                $outputsVariable = $func->forward($t, $x);
                $y = $g->mul($outputsVariable,$g->Variable(2));
                return $y;
            }
        );
        $loss = $K->scalar($y);
        $this->assertLessThan(1e-2,abs($loss-27.63));
        $mul = $y->creator();
        $dx = $mul->backward([$K->array(1.0)]);
        $lossfunc = $mul->inputs()[0]->creator();
        $dx = $lossfunc->backward([$dx[0]]);
        $dx = $dx[1];
        $dx = $K->ndarray($dx);
        //$dx = $y->creator()->backward([$K->array(1.0)]);
        //$dx = $dx[1];
        //$dx = $K->ndarray($dx);

        $x = $mo->array([
            [0.00001, 0.20000 , 0.79998],
            [0.79998, 0.00001 , 0.20000],
        ]);
        $t = $mo->array([2, 2],dtype:NDArray::int32);
        $this->assertTrue(
            $this->verifyGradient($mo,$nn,$K,$g,$func,$t,$x));
    }

    public function testReductionNone()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $func = $nn->losses()->SparseCategoricalCrossEntropy(reduction:'none');

        //
        // match
        //
        
        $x = $mo->array([
            [0.00000, 0.00000 , 1.00000],
            [0.99998, 0.00001 , 0.00001],
        ]);
        $t = $mo->array([2, 0],NDArray::int32);
        $copyx = $mo->copy($x);
        $copyt = $mo->copy($t);
        $t = $K->array($t);
        $x = $K->array($x);
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($func,$t, $x, $g) {
                $outputsVariable = $func->forward($t, $x);
                $y = $g->mul($outputsVariable,$g->Variable(2));
                return $y;
            }
        );
        $loss = $K->ndarray($y);
        $tt = $K->ndarray($t);
        $tx = $K->ndarray($x);
        $this->assertTrue($mo->la()->isclose(
            $loss,$mo->la()->array([-2.3841856e-7,3.9816299249e-5]),null,1e-4));
        $this->assertEquals($copyx->toArray(),$tx->toArray());
        $this->assertEquals($copyt->toArray(),$tt->toArray());
        $dx = $y->creator()->backward([$K->array([1.0,1.0])]);
        $dx = $dx[1];
        $dx = $K->ndarray($dx);
        //$this->assertLessThan(0.00001,abs($mo->sum($dx)));
        //$this->assertLessThan(0.001,abs(1-$dx[0][0])/6);
        $this->assertEquals($copyx->toArray(),$tx->toArray());
        $this->assertEquals($copyt->toArray(),$tt->toArray());

        $accuracy = $this->accuracy($nn,$func,$t,$x);
        $accuracy = $K->scalar($accuracy);

        //
        // unmatch
        //

        $x = $mo->array([
            [0.00000, 0.00000 , 1.00000],
            [0.99998, 0.00001 , 0.00001],
        ]);
        $t = $mo->array([1, 1],dtype:NDArray::int32);
        $t = $K->array($t);
        $x = $K->array($x);
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($func,$t, $x, $g) {
                $outputsVariable = $func->forward($t, $x);
                $y = $g->mul($outputsVariable,$g->Variable(2));
                return $y;
            }
        );
        $loss = $K->ndarray($y);
        $this->assertTrue($mo->la()->isclose(
            $loss,$mo->la()->array([32.236190,23.005950]),null,1e-4));
        $dx = $y->creator()->backward([$K->array([1.0,1.0])]);
        $dx = $dx[1];
        $dx = $K->ndarray($dx);

        $x = $mo->array([
            [0.00001, 0.20000 , 0.79998],
            [0.79998, 0.00001 , 0.20000],
        ]);
        $t = $mo->array([2, 2],dtype:NDArray::int32);
        $this->assertTrue(
            $this->verifyGradient($mo,$nn,$K,$g,$func,$t,$x));
    }

    public function testFromLogits()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
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
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($func,$t, $x, $g) {
                $outputsVariable = $func->forward($t, $x);
                $y = $g->mul($outputsVariable,$g->Variable(2));
                return $y;
            }
        );
        $loss = $K->scalar($y);
        $tt = $K->ndarray($t);
        $tx = $K->ndarray($x);
        $this->assertLessThan(0.00001,abs($loss));
        $this->assertEquals($copyx->toArray(),$tx->toArray());
        $this->assertEquals($copyt->toArray(),$tt->toArray());
        $dx = $y->creator()->backward([$K->array(1.0)]);
        $dx =  $dx[1];
        $tt = $K->ndarray($t);
        $tx = $K->ndarray($x);
        #$this->assertLessThan(0.00001,abs($mo->sum($dx)));
        $this->assertEquals($copyx->toArray(),$tx->toArray());
        $this->assertEquals($copyt->toArray(),$tt->toArray());

        $accuracy = $this->accuracy($nn,$func,$t,$x);
        $accuracy = $K->scalar($accuracy);

        $x = $mo->array([
            [-10.0, -10.0 , 10.0],
            [ 10.0, -10.0 ,-10.0],
        ]);
        $t = $mo->array([1, 1],dtype:NDArray::int32);
        $t = $K->array($t);
        $x = $K->array($x);
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($func,$t, $x, $g) {
                $outputsVariable = $func->forward($t, $x);
                $y = $g->mul($outputsVariable,$g->Variable(2));
                return $y;
            }
        );
        $loss = $K->scalar($y);
        $this->assertGreaterThan(10.0,abs($loss));

        $dx = $y->creator()->backward([$K->array(1.0)]);
        $dx = $dx[1];
        $dx = $K->ndarray($dx);
        //$this->assertLessThan(0.00001,abs($mo->sum($dx)));

        $x = $mo->array([
            [-2.0,  0.0, 2.0],
            [ 2.0, -2.0, 0.0],
        ]);
        $t = $mo->array([2, 2],dtype:NDArray::int32);
        $this->assertTrue(
            $this->verifyGradient($mo,$nn,$K,$g,$func,$t,$x,true));
    }

    public function testNoReductionAndFromLogits()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $func = $nn->losses()->SparseCategoricalCrossEntropy(from_logits:true,reduction:'none');

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
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($func,$t, $x, $g) {
                $outputsVariable = $func->forward($t, $x);
                $y = $g->mul($outputsVariable,$g->Variable(2));
                return $y;
            }
        );
        $loss = $K->ndarray($y);
        $tt = $K->ndarray($t);
        $tx = $K->ndarray($x);
        $this->assertTrue($mo->la()->isclose($loss,$mo->array([-2e-7,-2e-7])));
        $this->assertEquals($copyx->toArray(),$tx->toArray());
        $this->assertEquals($copyt->toArray(),$tt->toArray());

        $mul = $y->creator();
        $dx = $mul->backward([$K->array([1.0,1.0])]);
        $lossfunc = $mul->inputs()[0]->creator();
        $dx = $lossfunc->backward([$dx[0]]);
        $dx = $dx[1];
        $dx = $K->ndarray($dx);
        $tt = $K->ndarray($t);
        $tx = $K->ndarray($x);
        #$this->assertLessThan(0.00001,abs($mo->sum($dx)));
        $this->assertEquals($copyx->toArray(),$tx->toArray());
        $this->assertEquals($copyt->toArray(),$tt->toArray());

        $accuracy = $this->accuracy($nn,$func,$t,$x);
        $accuracy = $K->scalar($accuracy);

        $x = $mo->array([
            [-10.0, -10.0 , 10.0],
            [ 10.0, -10.0 ,-10.0],
        ]);
        $t = $mo->array([1, 1],dtype:NDArray::int32);
        $t = $K->array($t);
        $x = $K->array($x);
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($func,$t, $x, $g) {
                $outputsVariable = $func->forward($t, $x);
                $y = $g->mul($outputsVariable,$g->Variable(2));
                return $y;
            }
        );
        $loss = $K->ndarray($y);
        $this->assertTrue($mo->la()->isclose($loss,$mo->array([32.1953,32.1953])));

        $dx = $y->creator()->backward([$K->array([1.0,1.0])]);
        $dx = $dx[1];
        $dx = $K->ndarray($dx);
        //$this->assertLessThan(0.00001,abs($mo->sum($dx)));

        $x = $mo->array([
            [-2.0,  0.0, 2.0],
            [ 2.0, -2.0, 0.0],
        ]);
        $t = $mo->array([2, 2],dtype:NDArray::int32);
        $this->assertTrue(
            $this->verifyGradient($mo,$nn,$K,$g,$func,$t,$x,true));
    }

    public function testMultiBatchDimsLogitsFalseReductionSum()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();

        $func = $nn->losses()->SparseCategoricalCrossEntropy(from_logits:false,reduction:'sum');
        $x = $g->Variable($K->zeros([2,3,4]));
        $t = $g->Variable($K->zeros([2,3],dtype:NDArray::int32));
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

        $func = $nn->losses()->SparseCategoricalCrossEntropy(from_logits:true,reduction:'sum');
        $x = $g->Variable($K->zeros([2,3,4]));
        $t = $g->Variable($K->zeros([2,3],dtype:NDArray::int32));
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

        $func = $nn->losses()->SparseCategoricalCrossEntropy(from_logits:false,reduction:'none');
        $x = $g->Variable($K->zeros([2,3,4]));
        $t = $g->Variable($K->zeros([2,3],dtype:NDArray::int32));
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

        $func = $nn->losses()->SparseCategoricalCrossEntropy(from_logits:true,reduction:'none');
        $x = $g->Variable($K->zeros([2,3,4]));
        $t = $g->Variable($K->zeros([2,3],dtype:NDArray::int32));
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