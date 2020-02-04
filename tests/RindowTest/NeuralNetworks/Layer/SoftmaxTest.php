<?php
namespace RindowTest\NeuralNetworks\Layer\SoftmaxTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Layer\Softmax;


class Test extends TestCase
{
    public function testNormal()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $layer = new Softmax($backend);

        $layer->build($inputShape=[5]);
        $x = $mo->array([
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
        ]);
        $y = $layer->forward($x, $training=true);
        $this->assertEquals([
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
            ],$x->toArray());
        $this->assertEquals($backend->softmax($x)->toArray(),$y->toArray());

        $dout = $mo->array([
            [-0.5,-0.25,0.0,0.25,0.5],
            [-0.5,-0.25,0.0,0.25,0.5],
            [-0.5,-0.25,0.0,0.25,0.5],
            ]);
        $dx = $layer->backward($dout);
        $this->assertEquals([
            [-0.5,-0.25,0.0,0.25,0.5],
            [-0.5,-0.25,0.0,0.25,0.5],
            [-0.5,-0.25,0.0,0.25,0.5],
            ],$dout->toArray());
        $this->assertEquals($backend->dSoftmax($dout,$y)->toArray(),$dx->toArray());
    }
}
