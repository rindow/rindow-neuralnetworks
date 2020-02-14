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
    public function testBuilder()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $this->assertInstanceof(
            'Rindow\NeuralNetworks\Loss\MeanSquaredError',
            $nn->losses()->MeanSquaredError());
    }

    public function testOneHot()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $lossFunction = new MeanSquaredError($backend);

        $trues = $mo->array([
            [0.0, 0.0 , 1.0],
            [0.0, 1.0 , 0.0],
        ]);
        $predicts = $mo->array([
            [0.025, 0.025 , 0.95],
            [0.025, 0.95 , 0.025],
        ]);
        $loss = $lossFunction->loss($trues,$predicts);
        $this->assertTrue(0.01>abs(0.0-$loss));

        $dx = $lossFunction->differentiateLoss();
        $this->assertEquals($predicts->shape(),$dx->shape());
        $this->assertTrue($mo->asum($dx)<0.1);
    }
}
