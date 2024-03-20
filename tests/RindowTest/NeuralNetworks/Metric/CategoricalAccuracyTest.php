<?php
namespace RindowTest\NeuralNetworks\Metric\CategoricalAccuracyTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Plot\Plot;

class CategoricalAccuracyTest extends TestCase
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
        $metric = $nn->metrics->CategoricalAccuracy();

        $x = $K->array([
            [0.05, 0.95, 0],
            [0.1, 0.1, 0.8],
        ]);
        $t = $K->array([
            [0, 1, 0],
            [0, 0, 1],
        ]);
        $copyx = $K->copy($x);
        $copyt = $K->copy($t);

        $metric->update($t,$x);
        $accuracy = $metric->result();
        $this->assertLessThan(0.0001,abs(1-$accuracy));
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        $x = $K->array([
            [0.05, 0.95, 0],
            [0.1, 0.8, 0.1],
        ]);
        $metric->reset();
        $metric->update($t,$x);
        $accuracy = $metric->result();
        $this->assertLessThan(0.0001,abs(0.5-$accuracy));
    }

    public function testMultiBatch()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $metric = $nn->metrics->CategoricalAccuracy();

        /////////////////////////////////////
        $x = $K->zeros([2,3,4]);
        $t = $K->zeros([2,3,4]);

        $metric->reset();
        $metric->update($t,$x);
        $accuracy = $metric->result();
        $this->assertLessThan(0.0001,abs(1-$accuracy));

        /////////////////////////////////////
        $x = $K->array([
            [0.05, 0.95, 0],
            [0.1, 0.1, 0.8],
        ]);
        $t = $K->array([
            [0, 1, 0],
            [0, 0, 1],
        ]);

        $metric->reset();
        $metric->update($t,$x);
        $accuracy = $metric->result();
        $this->assertLessThan(0.0001,abs(1-$accuracy));
    }
}
