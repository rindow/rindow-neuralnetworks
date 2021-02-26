<?php
namespace RindowTest\NeuralNetworks\Layer\BatchNormalizationTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Layer\BatchNormalization;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;


class Test extends TestCase
{
    public function newBackend($mo)
    {
        $builder = new NeuralNetworks($mo);
        return $builder->backend();
    }

    public function testNormal()
    {
        $mo = new MatrixOperator();
        $K = $backend = $this->newBackend($mo);
        $layer = new BatchNormalization($backend);

        $layer->build($inputShape=[3]);
        [$beta,$gamma] = $layer->getParams();
        $this->assertEquals([3],$beta->shape());
        $this->assertEquals([3],$gamma->shape());
        [$dbeta,$dgamma] = $layer->getGrads();
        $this->assertEquals([3],$dbeta->shape());
        $this->assertEquals([3],$dgamma->shape());

        $gamma = $K->array([1.0, 1.0, 1.0]);
        $beta = $K->array([0.0, 0.0, 0.0]);
        $layer->build($inputShape=[3],[
            'sampleWeights'=>[$gamma,$beta],
        ]);

        // 3 input x 4 batch
        $x = $K->array([
            [0.0, 0.0 , 6.0],
            [0.0, 0.0 , 6.0],
            [0.0, 0.0 , 6.0],
            [0.0, 0.0 , 6.0],
        ]);
        $out = $layer->forward($x,$training=true);
        // 3 output x 4 batch
        $this->assertEquals([4,3],$out->shape());
        // 2 output x 4 batch
        $dout = $K->array([
            [0.0, 0.0, -0.5],
            [0.0, 0.0, -0.5],
            [0.0, 0.0, -0.5],
            [0.0, 0.0, -0.5],
        ]);
        $dx = $layer->backward($dout);
        // 3 input x 4 batch
        $this->assertEquals([4,3],$dx->shape());
    }

    public function testChannelsLast()
    {
        $mo = new MatrixOperator();
        $K = $backend = $this->newBackend($mo);
        $layer = new BatchNormalization($backend);

        $layer->build($inputShape=[2,2,3]);
        [$beta,$gamma] = $layer->getParams();
        $this->assertEquals([3],$beta->shape());
        $this->assertEquals([3],$gamma->shape());
        [$dbeta,$dgamma] = $layer->getGrads();
        $this->assertEquals([3],$dbeta->shape());
        $this->assertEquals([3],$dgamma->shape());

        // 4 batch x 2x2x3
        $x = $K->array([
            [[[1.0,2.0,3.0],[0.5,1.5,2.5]],[[1.5,2.5,3.5],[1.0,2.0,3.0]]],
            [[[0.5,0.5,0.5],[0.5,0.5,0.5]],[[0.5,0.5,0.5],[0.5,0.5,0.5]]],
            [[[0.5,0.5,0.5],[0.5,0.5,0.5]],[[0.5,0.5,0.5],[0.5,0.5,0.5]]],
            [[[0.5,0.5,0.5],[0.5,0.5,0.5]],[[0.5,0.5,0.5],[0.5,0.5,0.5]]],
        ]);
        $out = $layer->forward($x,$training=true);
        // 4 batch x 2x2 image x 3 channels
        $this->assertEquals([4,2,2,3],$out->shape());
        // 2 output x 4 batch
        $dout = $K->array([
            [[[1.0,2.0,3.0],[0.5,1.5,2.5]],[[1.5,2.5,3.5],[1.0,2.0,3.0]]],
            [[[0.5,0.5,0.5],[0.5,0.5,0.5]],[[0.5,0.5,0.5],[0.5,0.5,0.5]]],
            [[[0.5,0.5,0.5],[0.5,0.5,0.5]],[[0.5,0.5,0.5],[0.5,0.5,0.5]]],
            [[[0.5,0.5,0.5],[0.5,0.5,0.5]],[[0.5,0.5,0.5],[0.5,0.5,0.5]]],
        ]);
        $dx = $layer->backward($dout);
        // 4 batch x 2x2 image x 3 input x
        $this->assertEquals([4,2,2,3],$dx->shape());
    }


    public function testChannelsFirst()
    {
        $mo = new MatrixOperator();
        $K = $backend = $this->newBackend($mo);
        $layer = new BatchNormalization($backend,[
            'axis'=>1,
        ]);

        $layer->build($inputShape=[3,2,2]);
        [$beta,$gamma] = $layer->getParams();
        $this->assertEquals([3],$beta->shape());
        $this->assertEquals([3],$gamma->shape());
        [$dbeta,$dgamma] = $layer->getGrads();
        $this->assertEquals([3],$dbeta->shape());
        $this->assertEquals([3],$dgamma->shape());

        $gamma = $K->array([1.0, 1.0, 1.0]);
        $beta = $K->array([0.0, 0.0, 0.0]);

        // 4 batch x 3x2x2
        $x = $K->array([
            [[[1.0,0.5],[1.5,1.0]],[[2.0,1.5],[2.5,2.0]],[[3.0,2.5],[3.5,3.0]]],
            [[[1.0,0.5],[1.5,1.0]],[[2.0,1.5],[2.5,2.0]],[[3.0,2.5],[3.5,3.0]]],
            [[[1.0,0.5],[1.5,1.0]],[[2.0,1.5],[2.5,2.0]],[[3.0,2.5],[3.5,3.0]]],
            [[[1.0,0.5],[1.5,1.0]],[[2.0,1.5],[2.5,2.0]],[[3.0,2.5],[3.5,3.0]]],
        ]);
        $out = $layer->forward($x,$training=true);
        // 4 batch x 2x2 image x 3 channels
        $this->assertEquals([4,3,2,2],$out->shape());
        // 2 output x 4 batch
        $dout = $K->array([
            [[[1.0,0.5],[1.5,1.0]],[[2.0,1.5],[2.5,2.0]],[[3.0,2.5],[3.5,3.0]]],
            [[[1.0,0.5],[1.5,1.0]],[[2.0,1.5],[2.5,2.0]],[[3.0,2.5],[3.5,3.0]]],
            [[[1.0,0.5],[1.5,1.0]],[[2.0,1.5],[2.5,2.0]],[[3.0,2.5],[3.5,3.0]]],
            [[[1.0,0.5],[1.5,1.0]],[[2.0,1.5],[2.5,2.0]],[[3.0,2.5],[3.5,3.0]]],
        ]);
        $dx = $layer->backward($dout);
        // 4 batch x 2x2 image x 3 input x
        $this->assertEquals([4,3,2,2],$dx->shape());
    }
}
