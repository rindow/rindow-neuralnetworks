<?php
namespace RindowTest\NeuralNetworks\Layer\SoftmaxWithSparseCategoricalCrossEntropyTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Loss\SoftmaxWithSparseCategoricalCrossEntropy;
use Interop\Polite\Math\Matrix\NDArray;

class Test extends TestCase
{
    public function testWithLabel()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $layer = new SoftmaxWithSparseCategoricalCrossEntropy($backend);
        $layer->build([3]);

        $x = $mo->array([
            [0.0, 0.0 , 6.0],
            [0.0, 0.0 , 6.0],
        ]);
        $t = $mo->array([2, 2],NDArray::int64);
        $y = $layer->forward($x, $training=true);
        $loss = $layer->loss($t,$y);
        $accuracy = $layer->accuracy($t,$x);

        $this->assertTrue(0.01>abs(0.0-$loss));

        $dx = $layer->backward($layer->differentiateLoss());
        $this->assertTrue($mo->asum($dx)<0.01);

        $x = $mo->array([
            [0.0, 0.0 , 6.0],
            [0.0, 0.0 , 6.0],
        ]);
        $t = $mo->array([1, 1]);
        $y = $layer->forward($x, $training=true);
        $loss = $layer->loss($t,$y);
        $this->assertTrue(0.01>abs(6.0-$loss));

        $dx = $layer->backward($layer->differentiateLoss());
        $this->assertTrue(abs( 0.0-$dx[0][0])<0.01);
        $this->assertTrue(abs(-0.5-$dx[0][1])<0.01);
        $this->assertTrue(abs( 0.5-$dx[0][2])<0.01);
        $this->assertTrue(abs( 0.0-$dx[1][0])<0.01);
        $this->assertTrue(abs(-0.5-$dx[1][1])<0.01);
        $this->assertTrue(abs( 0.5-$dx[1][2])<0.01);
    }

/*
    public function testOneHot()
    {
        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);
        $softmaxWithLoss = $nn->layers()->SoftmaxWithLoss();

        $x = $mo->array([
            [0.0, 0.0 , 6.0],
            [0.0, 0.0 , 6.0],
        ]);
        $t = $mo->array([
            [0.0, 0.0 , 1.0],
            [0.0, 0.0 , 1.0],
        ]);
        $loss = $softmaxWithLoss->forward($x,$t);
        $this->assertTrue(0.01>abs(0.0-$loss));

        $dx = $softmaxWithLoss->backward();
        $this->assertTrue($mo->asum($dx)<0.01);

        $x = $mo->array([
            [0.0, 0.0 , 6.0],
            [0.0, 0.0 , 6.0],
        ]);
        $t = $mo->array([
            [0.0, 1.0 , 0.0],
            [0.0, 1.0 , 0.0],
        ]);
        $loss = $softmaxWithLoss->forward($x,$t);
        $this->assertTrue(0.01>abs(6.0-$loss));

        $dx = $softmaxWithLoss->backward();
        $this->assertTrue(abs( 0.0-$dx[0][0])<0.01);
        $this->assertTrue(abs(-0.5-$dx[0][1])<0.01);
        $this->assertTrue(abs( 0.5-$dx[0][2])<0.01);
        $this->assertTrue(abs( 0.0-$dx[1][0])<0.01);
        $this->assertTrue(abs(-0.5-$dx[1][1])<0.01);
        $this->assertTrue(abs( 0.5-$dx[1][2])<0.01);
    }
*/
}
