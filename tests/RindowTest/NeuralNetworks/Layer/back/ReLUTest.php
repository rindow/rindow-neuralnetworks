<?php
namespace RindowTest\NeuralNetworks\Layer\ReLUTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Layer\ReLU;

class Test extends TestCase
{
    public function testNormal()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $layer = new ReLU($backend);

        $layer->build($inputShape=[5]);

        $inputs = $mo->array([
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
        ]);
        $copyInputs = $mo->copy($inputs);
        $outputs = $layer->forward($inputs, $training=true);
        $this->assertEquals([4,5],$outputs->shape());
        $this->assertEquals(
            [0.0,0.0,0.0,0.5,1.0],
            $outputs[0]->toArray()
        );
        $this->assertEquals($copyInputs->toArray(),$inputs->toArray());


        $dOutputs = $mo->array([
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
        ]);
        $copydOutputs = $mo->copy($dOutputs);
        $dInputs = $layer->backward($dOutputs);
        $this->assertEquals([4,5],$dInputs->shape());
        $this->assertEquals(
            [0.0,0.0,0.0,0.5,1.0],
            $dInputs[0]->toArray()
        );
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());

    }
}
