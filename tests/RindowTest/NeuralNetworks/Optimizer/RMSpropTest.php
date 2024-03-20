<?php
namespace RindowTest\NeuralNetworks\Optimizer\RMSpropTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Optimizer\RMSprop;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Optimizer\Schedule\InverseTimeDecay;

class RMSpropTest extends TestCase
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
        $optimizer = new RMSprop($backend);

        $weight = $K->array([
            [0.1, 0.2], [0.1, 0.1], [0.2, 0.2]
        ]);
        $bias = $K->array([0.5, 0.5]);
        $params = [
            $K->copy($weight),
            $K->copy($bias),
            $K->copy($weight),
            $K->copy($bias)
        ];
        $dWeight = $K->array([
            [0.1, 0.2], [-0.1, -0.1], [0.2, 0.2]
        ]);
        $dBias = $K->array([0.5, 0.5]);
        $grads = [
            $K->copy($dWeight),
            $K->copy($dBias),
            $K->copy($dWeight),
            $K->copy($dBias)
        ];
        $optimizer->update($params,$grads);
        $this->assertTrue(
            0.001 < $K->scalar($K->asum($K->sub($params[0],
            $K->array([[0.099, 0.198], [0.101, 0.101], [0.198, 0.198]]))))
        );
        $this->assertTrue(
            0.001 < $K->scalar($K->asum($K->sub($params[1],
            $K->array([0.495, 0.495]))))
        );
        $this->assertTrue(
            0.001 < $K->scalar($K->asum($K->sub($params[2],
            $K->array([[0.099, 0.198], [0.101, 0.101], [0.198, 0.198]]))))
        );
        $this->assertTrue(
            0.001 < $K->scalar($K->asum($K->sub($params[3],
            $K->array([0.495, 0.495]))))
        );
    }

    public function testDecay()
    {
        $mo = new MatrixOperator();
        $K = $backend = $this->newBackend($mo);
        $optimizer = new RMSprop($backend,lr:0.001,decay:0.001);

        $weight = $K->array([
            [0.1, 0.2], [0.1, 0.1], [0.2, 0.2]
        ]);
        $bias = $K->array([0.5, 0.5]);
        $params = [
            $K->copy($weight),
            $K->copy($bias),
            $K->copy($weight),
            $K->copy($bias)
        ];
        $dWeight = $K->array([
            [0.1, 0.2], [-0.1, -0.1], [0.2, 0.2]
        ]);
        $dBias = $K->array([0.5, 0.5]);
        $grads = [
            $K->copy($dWeight),
            $K->copy($dBias),
            $K->copy($dWeight),
            $K->copy($dBias)
        ];
        for($i=0;$i<10;$i++) {
            $optimizer->update($params,$grads);
        }
        $this->assertTrue(true);
    }

    public function testSchedule()
    {
        $mo = new MatrixOperator();
        $K = $backend = $this->newBackend($mo);
        $schedule = new InverseTimeDecay(
            $init=0.001,$decaySteps=1,$decayRate=0.001);
        $optimizer = new RMSprop($backend,lr:$schedule);

        $weight = $K->array([
            [0.1, 0.2], [0.1, 0.1], [0.2, 0.2]
        ]);
        $bias = $K->array([0.5, 0.5]);
        $params = [
            $K->copy($weight),
            $K->copy($bias),
            $K->copy($weight),
            $K->copy($bias)
        ];
        $dWeight = $K->array([
            [0.1, 0.2], [-0.1, -0.1], [0.2, 0.2]
        ]);
        $dBias = $K->array([0.5, 0.5]);
        $grads = [
            $K->copy($dWeight),
            $K->copy($dBias),
            $K->copy($dWeight),
            $K->copy($dBias)
        ];
        for($i=0;$i<10;$i++) {
            $optimizer->update($params,$grads);
        }
        $this->assertTrue(true);
    }
}
