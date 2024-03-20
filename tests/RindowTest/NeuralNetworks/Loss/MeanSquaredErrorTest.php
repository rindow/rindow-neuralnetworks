<?php
namespace RindowTest\NeuralNetworks\Loss\MeanSquaredErrorTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Loss\MeanSquaredError;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;

class MeanSquaredErrorTest extends TestCase
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

    public function testBuilder()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $this->assertInstanceof(
            MeanSquaredError::class,
            $nn->losses()->MeanSquaredError());
    }

    public function testOneHot()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $lossFunction = $nn->losses()->MeanSquaredError();

        $predicts = $K->array([
            [0.025, 0.025 , 0.95],
            [0.025, 0.95 , 0.025],
        ]);
        $trues = $K->array([
            [0.0, 0.0 , 1.0],
            [0.0, 1.0 , 0.0],
        ]);
        $loss = $lossFunction->forward($trues,$predicts);
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($lossFunction,$trues, $predicts) {
                $outputsVariable = $lossFunction->forward($trues, $predicts);
                return $outputsVariable;
            }
        );
        $loss = $K->scalar($outputsVariable);
        $this->assertTrue(0.01>abs(0.0-$loss));

        $dx = $outputsVariable->creator()->backward([$K->array(1.0)]);
        $dx = $dx[1];
        $this->assertEquals($predicts->shape(),$dx->shape());
        $this->assertTrue($K->scalar($K->asum($dx))<0.1);
        $dx = $K->ndarray($dx);
        $this->assertTrue($mo->la()->isclose(
            $dx,$mo->la()->array([
                [ 0.00833333,  0.00833333, -0.01666667],
                [ 0.00833333, -0.01666667,  0.00833333]
            ]))
        );

        //
        // verifyGradient
        //

        $x = $K->array([
            [0.025, 0.025 , 0.95],
            [0.025, 0.95 , 0.025],
        ]);
        $t = $K->array([
            [0.0, 0.0 , 1.0],
            [0.0, 1.0 , 0.0],
        ]);
        $this->assertTrue(
            $this->verifyGradient($mo,$nn,$K,$g,$lossFunction,$t,$x));
    }

    public function testReductionNone()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $lossFunction = $nn->losses()->MeanSquaredError(reduction:'none');

        $predicts = $K->array([
            [0.025, 0.025 , 0.95],
            [0.025, 0.95 , 0.025],
        ]);
        $trues = $K->array([
            [0.0, 0.0 , 1.0],
            [0.0, 1.0 , 0.0],
        ]);
        $loss = $lossFunction->forward($trues,$predicts);
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($lossFunction,$trues, $predicts) {
                $outputsVariable = $lossFunction->forward($trues, $predicts);
                return $outputsVariable;
            }
        );
        $loss = $K->ndarray($outputsVariable);
        $this->assertTrue($mo->la()->isclose(
            $loss,$mo->la()->array([0.00125, 0.00125]))
        );

        $dx = $outputsVariable->creator()->backward([$K->array([1.0,1.0])]);
        $dx = $K->ndarray($dx[1]);
        $this->assertEquals($predicts->shape(),$dx->shape());
        $this->assertTrue($mo->la()->isclose(
            $dx,$mo->la()->array([
                [ 0.01666667,  0.01666667, -0.03333334],
                [ 0.01666667, -0.03333334,  0.01666667]
            ]))
        );
        //$this->assertTrue($K->scalar($K->asum($dx))<0.1);

        //
        // verifyGradient
        //

        $x = $K->array([
            [0.025, 0.025 , 0.95],
            [0.025, 0.95 , 0.025],
        ]);
        $t = $K->array([
            [0.0, 0.0 , 1.0],
            [0.0, 1.0 , 0.0],
        ]);
        $this->assertTrue(
            $this->verifyGradient($mo,$nn,$K,$g,$lossFunction,$t,$x));
    }

    public function testForwardOtherReductionNone()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $lossFunction = $nn->losses()->MeanSquaredError(reduction:'none');

        $predicts = $K->array([
            [0.1, 0.9],
            [0.4, 0.6],
            [0.5, 0.5],
        ]);
        $trues = $K->array([
            [0.2, 0.8],
            [0.2, 0.8],
            [0.5, 0.5],
        ]);
        $loss = $lossFunction->forward($trues,$predicts);
        $loss = $K->ndarray($loss);
        $trueLoss = $mo->array([0.01,0.04,0.0]);
        $this->assertTrue($mo->la()->isclose($loss,$trueLoss));
    }
}
