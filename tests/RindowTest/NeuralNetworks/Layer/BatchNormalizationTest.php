<?php
namespace RindowTest\NeuralNetworks\Layer\BatchNormalizationTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Layer\BatchNormalization;


class Test extends TestCase
{
    public function testNormal()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $layer = new BatchNormalization($backend);

        $layer->build($inputShape=[3]);
        [$beta,$gamma] = $layer->getParams();
        $this->assertEquals([3],$beta->shape());
        $this->assertEquals([3],$gamma->shape());
        [$dbeta,$dgamma] = $layer->getGrads();
        $this->assertEquals([3],$dbeta->shape());
        $this->assertEquals([3],$dgamma->shape());

        $gamma = $mo->array([1.0, 1.0, 1.0]);
        $beta = $mo->array([0.0, 0.0, 0.0]);
        $layer->build($inputShape=[3],[
            'sampleWeights'=>[$gamma,$beta],
        ]);

        // 3 input x 4 batch
        $x = $mo->array([
            [0.0, 0.0 , 6.0],
            [0.0, 0.0 , 6.0],
            [0.0, 0.0 , 6.0],
            [0.0, 0.0 , 6.0],
        ]);
        $out = $layer->forward($x,$training=true);
        // 3 output x 4 batch
        $this->assertEquals([4,3],$out->shape());
        // 2 output x 4 batch
        $dout = $mo->array([
            [0.0, 0.0, -0.5],
            [0.0, 0.0, -0.5],
            [0.0, 0.0, -0.5],
            [0.0, 0.0, -0.5],
        ]);
        $dx = $layer->backward($dout);
        // 3 input x 4 batch
        $this->assertEquals([4,3],$dx->shape());
    }
}
