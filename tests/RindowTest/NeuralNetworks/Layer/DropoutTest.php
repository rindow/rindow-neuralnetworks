<?php
namespace RindowTest\NeuralNetworks\Layer\DropoutTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Layer\Dropout;


class Test extends TestCase
{
    public function testNormal()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $fn = $backend;
        $layer = new Dropout($backend,0.5);
        $layer->build([]);

        $x = $mo->array([-1.0,-0.5,0.1,0.5,1.0]);
        $y = $layer->forward($x,$training=true);
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

        $dout = $mo->copy($x);
        $dx = $layer->backward($dout);
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
