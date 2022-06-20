<?php
namespace RindowTest\NeuralNetworks\Loss\MeanSquaredErrorTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Loss\MeanSquaredError;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;

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

    public function getPlotConfig()
    {
        return [
            'renderer.skipCleaning' => true,
            'renderer.skipRunViewer' => getenv('TRAVIS_PHP_VERSION') ? true : false,
        ];
    }

    public function testBuilder()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $this->assertInstanceof(
            'Rindow\NeuralNetworks\Loss\MeanSquaredError',
            $nn->losses()->MeanSquaredError());
    }

    public function testOneHot()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $lossFunction = new MeanSquaredError($K);

        $trues = $K->array([
            [0.0, 0.0 , 1.0],
            [0.0, 1.0 , 0.0],
        ]);
        $predicts = $K->array([
            [0.025, 0.025 , 0.95],
            [0.025, 0.95 , 0.025],
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
    }
}
