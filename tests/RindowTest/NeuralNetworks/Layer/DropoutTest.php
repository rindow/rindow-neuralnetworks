<?php
namespace RindowTest\NeuralNetworks\Layer\DropoutTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Layer\Dropout;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;


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

    public function testNormal()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();

        $layer = new Dropout($K,0.5);
        $x = $K->array([-1.0,-0.5,0.1,0.5,1.0]);

        $inputs = $g->Variable($x);
        $layer->build($inputs);
        $this->assertEquals([],$layer->outputShape());

        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$x) {
                $outputsVariable = $layer->forward($x, $training=true);
                return $outputsVariable;
            }
        );
        $y = $K->ndarray($outputsVariable);
        $fn = $K;
        $this->assertTrue($fn->equalTest($y[0],-1.0)||
                          $fn->equalTest($y[0],0.0));
        $this->assertTrue($fn->equalTest($y[1],-0.5)||
                          $fn->equalTest($y[1],0.0));
        $this->assertTrue($fn->equalTest($y[2],0.1)||
                          $fn->equalTest($y[2],0.0));
        $this->assertTrue($fn->equalTest($y[3],0.5)||
                          $fn->equalTest($y[3],0.0));
        $this->assertTrue($fn->equalTest($y[4],1.0)||
                          $fn->equalTest($y[4],0.0));

        $dout = $K->copy($x);
        [$dx] = $outputsVariable->creator()->backward([$dout]);
        $dx = $K->ndarray($dx);
        $this->assertTrue($fn->equalTest($dx[0],-1.0)||
                          $fn->equalTest($dx[0],0.0));
        $this->assertTrue($fn->equalTest($dx[1],-0.5)||
                          $fn->equalTest($dx[1],0.0));
        $this->assertTrue($fn->equalTest($dx[2],0.1)||
                          $fn->equalTest($dx[2],0.0));
        $this->assertTrue($fn->equalTest($dx[3],0.5)||
                          $fn->equalTest($dx[3],0.0));
        $this->assertTrue($fn->equalTest($dx[4],1.0)||
                          $fn->equalTest($dx[4],0.0));
    }
}
