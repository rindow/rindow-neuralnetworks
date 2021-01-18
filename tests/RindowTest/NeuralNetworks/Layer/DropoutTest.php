<?php
namespace RindowTest\NeuralNetworks\Layer\DropoutTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Layer\Dropout;
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
        $fn = $backend;
        $layer = new Dropout($backend,0.5);
        $layer->build([]);

        $x = $K->array([-1.0,-0.5,0.1,0.5,1.0]);
        $y = $layer->forward($x,$training=true);
        $y = $K->ndarray($y);
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
        $dx = $layer->backward($dout);
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
