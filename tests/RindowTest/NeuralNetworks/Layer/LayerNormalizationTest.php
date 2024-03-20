<?php
namespace RindowTest\NeuralNetworks\Layer\LayerNormalizationTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Layer\LayerNormalization;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

class LayerNormalizationTest extends TestCase
{
    public function newMatrixOperator()
    {
        return new MatrixOperator();
    }

    public function newNeuralNetworks($mo)
    {
        return new NeuralNetworks($mo);
    }

    public function testNormal()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();

        $layer = new LayerNormalization($K);

        // 3 input x 4 batch
        $x = $K->array([
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0],
            [3.0, 2.0, 1.0],
        ]);

        $inputs = $g->Variable($x);
        $layer->build($inputs);
        [$beta,$gamma] = $layer->getParams();
        $this->assertEquals([3],$beta->shape());
        $this->assertEquals([3],$gamma->shape());
        [$dbeta,$dgamma] = $layer->getGrads();
        $this->assertEquals([3],$dbeta->shape());
        $this->assertEquals([3],$dgamma->shape());

        $gamma = $K->array([1.0, 1.0, 1.0]);
        $beta = $K->array([0.0, 0.0, 0.0]);
        $layer->build($inputs, sampleWeights:[$gamma,$beta]);

        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$x) {
                $outputsVariable = $layer->forward($x, training:true);
                return $outputsVariable;
            }
        );
        $out = $K->ndarray($outputsVariable);
        // 3 output x 4 batch
        $this->assertEquals([4,3],$out->shape());
        $this->assertTrue($mo->la()->isclose($mo->la()->array(
            [[-1.2238274,  0.0,  1.2238274],
             [-1.2238274,  0.0,  1.2238274],
             [ 1.2238274,  0.0, -1.2238274],
             [ 1.2238274,  0.0, -1.2238274],]
        ), $out));
        // 2 output x 4 batch
        $dout = $K->array([
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0],
            [1.0, 0.5, 0.0],
            [1.0, 0.5, 0.0],
        ]);
        [$dx] = $outputsVariable->creator()->backward([$dout]);
        $dx = $K->ndarray($dx);
        $this->assertTrue($mo->la()->isclose($mo->la()->array(
            [[-0.00069,  0.15298,  0.30664],
             [-0.00069,  0.15298,  0.30664],
             [ 0.30664,  0.15298, -0.00069],
             [ 0.30664,  0.15298, -0.00069]]
        ), $dx, $r=1e-1));
        // 3 input x 4 batch
        $this->assertEquals([4,3],$dx->shape());

        $this->assertCount(2,$layer->variables());
        $this->assertCount(2,$layer->trainableVariables());

    }

    public function testClone()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();

        $layer = new LayerNormalization($K);
        $this->assertCount(2,$layer->variables());
        $this->assertCount(2,$layer->trainableVariables());

        $layer2 = clone $layer;
        $this->assertCount(2,$layer2->variables());
        $this->assertCount(2,$layer2->trainableVariables());
    }

    public function testChannelsLast()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new LayerNormalization($K);
        // 4 batch x 2x2x3
        $x = $K->array([
            [[[1.0,2.0,3.0],[0.5,1.5,2.5]],[[1.5,2.5,3.5],[1.0,2.0,3.0]]],
            [[[0.5,0.5,0.5],[0.5,0.5,0.5]],[[0.5,0.5,0.5],[0.5,0.5,0.5]]],
            [[[0.5,0.5,0.5],[0.5,0.5,0.5]],[[0.5,0.5,0.5],[0.5,0.5,0.5]]],
            [[[0.5,0.5,0.5],[0.5,0.5,0.5]],[[0.5,0.5,0.5],[0.5,0.5,0.5]]],
        ]);

        $inputs = $g->Variable($x);
        $layer->build($inputs);
        [$beta,$gamma] = $layer->getParams();
        $this->assertEquals([3],$beta->shape());
        $this->assertEquals([3],$gamma->shape());
        [$dbeta,$dgamma] = $layer->getGrads();
        $this->assertEquals([3],$dbeta->shape());
        $this->assertEquals([3],$dgamma->shape());

        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$x) {
                $outputsVariable = $layer->forward($x, training:true);
                return $outputsVariable;
            }
        );
        $out = $K->ndarray($outputsVariable);
        // 4 batch x 2x2 image x 3 channels
        $this->assertEquals([4,2,2,3],$out->shape());
        // 2 output x 4 batch
        $dout = $K->array([
            [[[1.0,2.0,3.0],[0.5,1.5,2.5]],[[1.5,2.5,3.5],[1.0,2.0,3.0]]],
            [[[0.5,0.5,0.5],[0.5,0.5,0.5]],[[0.5,0.5,0.5],[0.5,0.5,0.5]]],
            [[[0.5,0.5,0.5],[0.5,0.5,0.5]],[[0.5,0.5,0.5],[0.5,0.5,0.5]]],
            [[[0.5,0.5,0.5],[0.5,0.5,0.5]],[[0.5,0.5,0.5],[0.5,0.5,0.5]]],
        ]);
        [$dx] = $outputsVariable->creator()->backward([$dout]);
        // 4 batch x 2x2 image x 3 input x
        $this->assertEquals([4,2,2,3],$dx->shape());
    }


    public function testChannelsFirst()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new LayerNormalization($K,
            axis:1,
        );
        // 4 batch x 3x2x2
        $x = $K->array([
            [[[1.0,0.5],[1.5,1.0]],[[2.0,1.5],[2.5,2.0]],[[3.0,2.5],[3.5,3.0]]],
            [[[1.0,0.5],[1.5,1.0]],[[2.0,1.5],[2.5,2.0]],[[3.0,2.5],[3.5,3.0]]],
            [[[1.0,0.5],[1.5,1.0]],[[2.0,1.5],[2.5,2.0]],[[3.0,2.5],[3.5,3.0]]],
            [[[1.0,0.5],[1.5,1.0]],[[2.0,1.5],[2.5,2.0]],[[3.0,2.5],[3.5,3.0]]],
        ]);

        $inputs = $g->Variable($x);
        $layer->build($inputs);
        [$beta,$gamma] = $layer->getParams();
        $this->assertEquals([3],$beta->shape());
        $this->assertEquals([3],$gamma->shape());
        [$dbeta,$dgamma] = $layer->getGrads();
        $this->assertEquals([3],$dbeta->shape());
        $this->assertEquals([3],$dgamma->shape());

        $gamma = $K->array([1.0, 1.0, 1.0]);
        $beta = $K->array([0.0, 0.0, 0.0]);

        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$x) {
                $outputsVariable = $layer->forward($x, training:true);
                return $outputsVariable;
            }
        );
        $out = $K->ndarray($outputsVariable);
        // 4 batch x 2x2 image x 3 channels
        $this->assertEquals([4,3,2,2],$out->shape());
        // 2 output x 4 batch
        $dout = $K->array([
            [[[1.0,0.5],[1.5,1.0]],[[2.0,1.5],[2.5,2.0]],[[3.0,2.5],[3.5,3.0]]],
            [[[1.0,0.5],[1.5,1.0]],[[2.0,1.5],[2.5,2.0]],[[3.0,2.5],[3.5,3.0]]],
            [[[1.0,0.5],[1.5,1.0]],[[2.0,1.5],[2.5,2.0]],[[3.0,2.5],[3.5,3.0]]],
            [[[1.0,0.5],[1.5,1.0]],[[2.0,1.5],[2.5,2.0]],[[3.0,2.5],[3.5,3.0]]],
        ]);
        [$dx] = $outputsVariable->creator()->backward([$dout]);
        // 4 batch x 2x2 image x 3 input x
        $this->assertEquals([4,3,2,2],$dx->shape());
    }
}
