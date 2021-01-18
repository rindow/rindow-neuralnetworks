<?php
namespace RindowTest\NeuralNetworks\Optimizer\SGDTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Optimizer\SGD;
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
        $optimizer = new SGD($backend);

        $weight = $K->array([
            [0.1, 0.2], [0.1, 0.1], [0.2, 0.2]
        ]);
        $bias = $K->array([0.5, 0.5]);
        $params = [
            $K->copy($weight),
            $K->copy($bias),
            $K->copy($weight),
            $K->copy($bias),
        ];
        $dWeight = $K->array([
            [0.1, 0.2], [-0.1, -0.1], [0.2, 0.2]
        ]);
        $dBias = $K->array([0.5, 0.5]);
        $grads = [
            $K->copy($dWeight),
            $K->copy($dBias),
            $K->copy($dWeight),
            $K->copy($dBias),
        ];
        $optimizer->update($params,$grads);
        $this->assertTrue($fn->equalTest(
            $K->array([[0.099, 0.198], [0.101, 0.101], [0.198, 0.198]]),
            $params[0]
        ));
        $this->assertTrue($fn->equalTest(
            $K->array([0.495, 0.495]),
            $params[1]
        ));
        $this->assertTrue($fn->equalTest(
            $K->array([[0.099, 0.198], [0.101, 0.101], [0.198, 0.198]]),
            $params[2]
        ));
        $this->assertTrue($fn->equalTest(
            $K->array([0.495, 0.495]),
            $params[3]
        ));
    }
}
