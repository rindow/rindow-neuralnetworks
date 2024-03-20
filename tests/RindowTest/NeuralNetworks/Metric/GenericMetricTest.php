<?php
namespace RindowTest\NeuralNetworks\Metric\GenericMetricTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Plot\Plot;

class GenericMetricTest extends TestCase
{
    public function newMatrixOperator()
    {
        return new MatrixOperator();
    }

    public function newNeuralNetworks($mo)
    {
        return new NeuralNetworks($mo);
    }

    public function testDefault()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $func = function (NDArray $true, NDArray $predicts) use ($K)
        {
            return $K->scalar($K->sum($K->sub($true,$predicts)));
        };
        $metric = $nn->metrics->GenericMetric($func);

        $t = $K->array([
            [0.0], [0.0] , [0.0],
        ]);
        $p = $K->array([
            [0.0], [0.0] , [-1.0],
        ]);

        $metric->update($t,$p);
        $this->assertEquals(1.0,$metric->result());

        $p = $K->array([
            [0.0], [0.0] , [-2.0],
        ]);
        $metric->update($t,$p);
        $this->assertEquals(1.5,$metric->result());

        $metric->reset();
        $this->assertEquals(0.0,$metric->result());

        $p = $K->array([
            [0.0], [0.0] , [-2.0],
        ]);
        $metric->update($t,$p);
        $this->assertEquals(2.0,$metric->result());
    }
}
