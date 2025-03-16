<?php
namespace RindowTest\NeuralNetworks\Layer\LayerNormalizationTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
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
        //echo "============= testNormal ============================\n";
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
        //echo $mo->toString($dx,indent:true)."\n";
        $this->assertTrue($mo->la()->isclose($mo->la()->array([
            [-0.00091649,  0.0,          0.00091649],
            [-0.00091649,  0.0,          0.00091649],
            [ 0.00091649,  0.0,         -0.00091649],
            [ 0.00091649,  0.0,         -0.00091649],
        ]), $dx ));
        // 3 input x 4 batch
        $this->assertEquals([4,3],$dx->shape());

        $this->assertCount(2,$layer->variables());
        $this->assertCount(2,$layer->trainableVariables());

    }

    public function testClone()
    {
        //echo "============= testClone ============================\n";
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
        //echo "============= testChannelsLast ============================\n";
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new LayerNormalization($K);
        // 4 batch x 2x2x3
        // (batch,height,width,color)
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
        //echo "============= testChannelsFirst ============================\n";
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new LayerNormalization($K,
            axis:1,
        );
        // 4 batch x 3x2x2
        // (batch,color,height,width)
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

    public function testFloatingSequenceLength()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new LayerNormalization($K);

        //
        //  build and first call
        //
        // 4batch x 5seq x 2feature
        //echo "==========first===============\n";
        $shape = [4,5,2];
        $x = $K->array($mo->la()->range(start:1,limit:1+array_product($shape),dtype:NDArray::float32)
                ->reshape($shape));
        //echo "x=".$mo->toString($x,indent:true)."\n";
        $inputs = $g->Variable($x);
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$inputs) {
                $outputsVariable = $layer->forward($inputs, training:true);
                return $outputsVariable;
            }
        );
        $grads = $tape->gradient($outputsVariable,$inputs);
        
        [$beta,$gamma] = $layer->trainableVariables();
        $this->assertEquals([2],$beta->shape());
        $this->assertEquals([2],$gamma->shape());
        [$dbeta,$dgamma] = $layer->getGrads();
        $this->assertEquals([2],$dbeta->shape());
        $this->assertEquals([2],$dgamma->shape());

        //
        // change sequence length and second call
        //
        //echo "==========second===============\n";
        $layer->setShapeInspection(false);
        // 4batch x 3seq x 2feature
        $shape = [4,3,2];
        $x = $K->array($mo->la()->range(start:1,limit:1+array_product($shape),dtype:NDArray::float32)
                ->reshape($shape));
        $inputs = $g->Variable($x);

        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$inputs) {
                $outputsVariable = $layer->forward($inputs, training:true);
                return $outputsVariable;
            }
        );
        $out = $K->ndarray($outputsVariable);
        // 4batch x 3seq x 2feature
        $this->assertEquals([4,3,2],$out->shape());
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->array([
               [[-0.99800587,  0.99800587],
                [-0.99800587,  0.99800587],
                [-0.99800587,  0.99800587]],
                
               [[-0.99800587,  0.99800587],
                [-0.99800587,  0.99800587],
                [-0.99800587,  0.99800587]],
                
               [[-0.99800587,  0.99800587],
                [-0.99800587,  0.99800587],
                [-0.99800587,  0.99800587]],
                
               [[-0.99800587,  0.99800587],
                [-0.99800587,  0.99800587],
                [-0.99800587,  0.99800587]],
            ]),
            $K->ndarray($out),
            //debug:true,
        ));

        // 4batch x 3seq x 2feature
        $dout = $x;
        //echo "dout=".$mo->toString($dout,indent:true)."\n";
        [$dx] = $outputsVariable->creator()->backward([$dout]);
        // 4 batch x 2x2 image x 3 input x
        //echo $mo->toString($dx,format:'%13.8f',indent:true)."\n";
        //echo $mo->la()->sum($K->ndarray($dx))."\n";
        $this->assertEquals([4,3,2],$dx->shape());
        $this->assertTrue($mo->la()->isclose(
            $mo->la()->array([
                [[-0.00397633,  0.00397633],
                 [-0.00397633,  0.00397633],
                 [-0.00397633,  0.00397633]],
                [[-0.00397633,  0.00397633],
                 [-0.00397633,  0.00397633],
                 [-0.00397633,  0.00397633]],
                [[-0.00397633,  0.00397633],
                 [-0.00397633,  0.00397633],
                 [-0.00397633,  0.00397633]],
                [[-0.00397633,  0.00397633],
                 [-0.00397633,  0.00397633],
                 [-0.00397633,  0.00397633]],
            ]),
            $K->ndarray($dx),
            rtol:1e-3,
            //atol:1e-7,
            //debug:true,
        ));

        $this->assertTrue($mo->la()->isclose(
            $mo->la()->array([
                -143.71284,  155.6889
            ]),
            $K->ndarray($dgamma),
        ));

        $this->assertTrue($mo->la()->isclose(
            $mo->la()->array([
                144.0, 156.0,
            ]),
            $K->ndarray($dbeta),
        ));

    }

}
