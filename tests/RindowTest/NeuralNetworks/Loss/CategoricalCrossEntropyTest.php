<?php
namespace RindowTest\NeuralNetworks\Loss\CategoricalCrossEntropyTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Loss\CategoricalCrossEntropy;
use Rindow\NeuralNetworks\Metric\MetricCatalog;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;

class CategoricalCrossEntropyTest extends TestCase
{
    public function newMatrixOperator()
    {
        return new MatrixOperator();
    }

    public function newNeuralNetworks($mo)
    {
        return new NeuralNetworks($mo);
    }

    public function getPlotConfig()
    {
        return [
            'renderer.skipCleaning' => true,
            'renderer.skipRunViewer' => getenv('PLOT_RENDERER_SKIP') ? true : false,
        ];
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

    public function testBuilder()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $this->assertInstanceof(
            CategoricalCrossEntropy::class,
            $nn->losses()->CategoricalCrossEntropy());
    }

    public function testDefault()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $func = $nn->losses()->CategoricalCrossEntropy();
/*
        $x = $K->array([
            [0.0, 0.0 , 6.0],
            [0.0, 0.0 , 6.0],
        ]);
        $t = $K->array([
            [0.0, 0.0 , 1.0],
            [0.0, 0.0 , 1.0],
        ]);
        $n = $x->shape()[0];

        $y = $backend->softmax($x);
        $loss = $func->forward($t,$y);
        $accuracy = $this->accuracy($nn,$func,$t,$x);

        $this->assertLessThan(0.01,abs(0.0-$loss));

        $dx = $backend->dsoftmax($func->backward(1.0),$y);
        #$this->assertLessThan(0.0001, $K->scalar($K->asum($K->sub($K->sub($y,$dx),$t))));
*/
        #$x = $K->array([
        #    [0.0, 0.0 , 6.0],
        #    [0.0, 0.0 , 6.0],
        #]);
        #$t = $K->array([
        #    [0.0, 1.0 , 0.0],
        #    [0.0, 1.0 , 0.0],
        #]);
        $x = $K->array([
            [0.05, 0.95, 0],
            [0.1, 0.8, 0.1],
        ]);
        $t = $K->array([
            [0, 1, 0],
            [0, 0, 1],
        ]);

        $x = $K->softmax($x);
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($func,$t, $x) {
                $outputsVariable = $func->forward($t, $x);
                return $outputsVariable;
            }
        );
        $loss = $K->scalar($outputsVariable);
        $this->assertLessThan(0.01,abs(0.9868951-$loss));

        $dx = $outputsVariable->creator()->backward([$K->array(1.0)]);
        $dx = $K->dSoftmax($dx[1],$x);
        #$this->assertLessThan(0.001,$K->scalar($K->asum($K->sub($K->sub($x,$dx),$t))));
        $this->assertTrue($mo->la()->isclose(
            $mo->array([[ 0.11335728, -0.22118606,  0.10782879],
                        [ 0.12457169,  0.25085658, -0.3754283 ]]),
            $K->ndarray($dx)));


        $x = $K->array([
            [0.05, 0.95, 0], [0.1, 0.8, 0.1]
        ]);
        $x = $K->softmax($x);
        $t = $K->array([
            [0, 1, 0], [0, 0, 1]
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
        $func = $nn->losses()->CategoricalCrossEntropy(reduction:'none');

        $x = $K->array([
            [0.05, 0.95, 0],
            [0.1, 0.8, 0.1],
        ]);
        $t = $K->array([
            [0, 1, 0],
            [0, 0, 1],
        ]);

        $x = $K->softmax($x);
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($func,$t, $x) {
                $outputsVariable = $func->forward($t, $x);
                return $outputsVariable;
            }
        );
        $loss = $K->ndarray($outputsVariable);
        $this->assertTrue($mo->la()->isclose(
            $loss,$mo->la()->array([0.5840635, 1.3897266]),null,1e-4)
        );

        $dx = $outputsVariable->creator()->backward([$K->array([1.0,1.0])]);
        $dx = $K->dSoftmax($dx[1],$x);
        #$this->assertLessThan(0.001,$K->scalar($K->asum($K->sub($K->sub($x,$dx),$t))));
        $this->assertTrue($mo->la()->isclose(
            $mo->array([[ 0.22671457, -0.4423721 ,  0.21565758],
                        [ 0.24914338,  0.50171316, -0.7508566 ]]),
            $K->ndarray($dx)));

        $x = $K->array([
            [0.05, 0.95, 0], [0.1, 0.8, 0.1]
        ]);
        $x = $K->softmax($x);
        $t = $K->array([
            [0, 1, 0], [0, 0, 1]
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
        $func = $nn->losses()->CategoricalCrossEntropy(from_logits:true);
/*
        $x = $K->array([
            [0.0, 0.0 , 6.0],
            [0.0, 0.0 , 6.0],
        ]);
        $t = $K->array([
            [0.0, 0.0 , 1.0],
            [0.0, 0.0 , 1.0],
        ]);
        $y = $backend->softmax($x);

        $loss = $func->forward($t,$x);
        $this->assertLessThan(0.01,abs(0.0-$loss));

        $dx = $func->backward(1.0);
        $this->assertLessThan(0.0001,$K->scalar($K->asum($K->sub($K->sub($y,$dx),$t))));
*/
        #$x = $K->array([
        #    [0.0, 0.0 , 6.0],
        #    [0.0, 0.0 , 6.0],
        #]);
        #$t = $K->array([
        #    [0.0, 1.0 , 0.0],
        #    [0.0, 1.0 , 0.0],
        #]);
        $x = $K->array([
            [0.05, 0.95, 0],
            [0.1, 0.8, 0.1],
        ]);
        $t = $K->array([
            [0, 1, 0],
            [0, 0, 1],
        ]);
        //$y = $backend->softmax($x);

        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($func,$t, $x) {
                $outputsVariable = $func->forward($t, $x);
                return $outputsVariable;
            }
        );
        $loss = $K->scalar($outputsVariable);
        $this->assertLessThan(0.01,abs(0.9868951-$loss));

        $dx = $outputsVariable->creator()->backward([$K->array(1.0)]);
        $dx = $dx[1];
        #$this->assertLessThan(0.0001,$K->scalar($K->asum($K->sub($K->sub($y,$dx),$t))));
        $this->assertTrue($mo->la()->isclose(
            $mo->array([[ 0.11335728, -0.22118606,  0.10782879],
                        [ 0.12457169,  0.25085658, -0.3754283 ]]),
            $K->ndarray($dx)));

        $x = $K->array([
            [0.05, 0.95, 0], [0.1, 0.8, 0.1]
        ]);
        $x = $K->softmax($x);
        $t = $K->array([
            [0, 1, 0], [0, 0, 1]
        ]);
        $this->assertTrue(
            $this->verifyGradient($mo,$nn,$K,$g,$func,$t,$x));
    }

    public function testNoReductionAndFromLogits()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $func = $nn->losses()->CategoricalCrossEntropy(from_logits:true,reduction:'none');

        $x = $K->array([
            [0.05, 0.95, 0],
            [0.1, 0.8, 0.1],
        ]);
        $t = $K->array([
            [0, 1, 0],
            [0, 0, 1],
        ]);
        //$y = $backend->softmax($x);

        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($func,$t, $x) {
                $outputsVariable = $func->forward($t, $x);
                return $outputsVariable;
            }
        );
        $loss = $K->ndarray($outputsVariable);
        $this->assertTrue($mo->la()->isclose(
            $loss,$mo->la()->array([0.5840635, 1.3897266]),null,1e-4)
        );
        //$this->assertLessThan(0.01,abs(0.9868951-$loss));

        $dx = $outputsVariable->creator()->backward([$K->array([1.0,1.0])]);
        $dx = $dx[1];
        #$this->assertLessThan(0.0001,$K->scalar($K->asum($K->sub($K->sub($y,$dx),$t))));
        $this->assertTrue($mo->la()->isclose(
            $mo->array([[ 0.22671457, -0.4423721 ,  0.21565758],
                        [ 0.24914338,  0.50171316, -0.7508566 ]]),
            $K->ndarray($dx)));

        $x = $K->array([
            [0.05, 0.95, 0], [0.1, 0.8, 0.1]
        ]);
        $x = $K->softmax($x);
        $t = $K->array([
            [0, 1, 0], [0, 0, 1]
        ]);
        $this->assertTrue(
            $this->verifyGradient($mo,$nn,$K,$g,$func,$t,$x));
    }

    public function testMultiBatchDimsLogitsFalseReductionSum()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();

        $func = $nn->losses()->CategoricalCrossEntropy(from_logits:false,reduction:'sum');
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

        $func = $nn->losses()->CategoricalCrossEntropy(from_logits:true,reduction:'sum');
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

        $func = $nn->losses()->CategoricalCrossEntropy(from_logits:false,reduction:'none');
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

        $func = $nn->losses()->CategoricalCrossEntropy(from_logits:true,reduction:'none');
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
