<?php
namespace RindowTest\NeuralNetworks\Optimizer\AdamTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Optimizer\Adam;

class Test extends TestCase
{
    public function testNormal()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $optimizer = new Adam($backend);

        $weight = $mo->array([
            [0.1, 0.2], [0.1, 0.1], [0.2, 0.2]
        ]);
        $bias = $mo->array([0.5, 0.5]);
        $params = [
            $mo->copy($weight),
            $mo->copy($bias),
            $mo->copy($weight),
            $mo->copy($bias)
        ];
        $dWeight = $mo->array([
            [0.1, 0.2], [-0.1, -0.1], [0.2, 0.2]
        ]);
        $dBias = $mo->array([0.5, 0.5]);
        $grads = [
            $mo->copy($dWeight),
            $mo->copy($dBias),
            $mo->copy($dWeight),
            $mo->copy($dBias)
        ];
        $optimizer->update($params,$grads);
        $this->assertTrue(
            0.001 < $mo->asum($mo->op($params[0],'-',
            $mo->array([[0.099, 0.198], [0.101, 0.101], [0.198, 0.198]])))
        );
        $this->assertTrue(
            0.001 < $mo->asum($mo->op($params[1],'-',
            $mo->array([0.495, 0.495])))
        );
        $this->assertTrue(
            0.001 < $mo->asum($mo->op($params[2],'-',
            $mo->array([[0.099, 0.198], [0.101, 0.101], [0.198, 0.198]])))
        );
        $this->assertTrue(
            0.001 < $mo->asum($mo->op($params[3],'-',
            $mo->array([0.495, 0.495])))
        );
    }
}
