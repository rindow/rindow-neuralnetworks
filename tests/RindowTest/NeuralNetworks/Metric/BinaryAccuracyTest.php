<?php
namespace RindowTest\NeuralNetworks\Metric\BinaryAccuracyTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Plot\Plot;

class BinaryAccuracyTest extends TestCase
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
        $metric = $nn->metrics->BinaryAccuracy();

        $x = $K->array([
            [0.1], [0.1] , [0.8],
        ],NDArray::float32);
        $t = $K->array([
            [0], [0] , [1],
        ],NDArray::int32);
        $copyx = $K->copy($x);
        $copyt = $K->copy($t);

        $metric->update($t,$x);
        $accuracy = $metric->result();
        $this->assertLessThan(0.0001,abs(1-$accuracy));
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        $x = $K->array([
            [0.1], [8.0] , [0.1],
        ],NDArray::float32);
        $metric->reset();
        $metric->update($t,$x);
        $accuracy = $metric->result();
        $this->assertLessThan(0.0001,abs(0.333333-$accuracy));
    }

    public function testThreshold()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $metric = $nn->metrics->BinaryAccuracy(threshold:0.05);

        $x = $K->array([
            [0.01], [0.02] , [0.08],
        ],NDArray::float32);
        $t = $K->array([
            [0], [0] , [1],
        ],NDArray::int32);
        $copyx = $K->copy($x);
        $copyt = $K->copy($t);

        $metric->update($t,$x);
        $accuracy = $metric->result();
        $this->assertLessThan(0.0001,abs(1-$accuracy));
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        $x = $K->array([
            [0.01], [0.08] , [0.01],
        ],NDArray::float32);
        $metric->reset();
        $metric->update($t,$x);
        $accuracy = $metric->result();
        $this->assertLessThan(0.0001,abs(0.333333-$accuracy));
    }

    public function testMultiBatch()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $metric = $nn->metrics->BinaryAccuracy();

        /////////////////////////////////////
        $x = $K->array([
            [0.1], [0.1] , [0.8],
        ],NDArray::float32);
        $t = $K->array([
            0, 0 , 1,
        ],NDArray::int32);

        $metric->reset();
        $metric->update($t,$x);
        $accuracy = $metric->result();
        $this->assertLessThan(0.0001,abs(1-$accuracy));

        /////////////////////////////////////
        $x = $K->array([
            [[0.1], [0.1] , [0.8]],
            [[0.1], [0.1] , [0.8]],
        ],NDArray::float32);
        $t = $K->array([
            [0, 0 , 1],
            [0, 0 , 1],
        ],NDArray::int32);

        $metric->reset();
        $metric->update($t,$x);
        $accuracy = $metric->result();
        $this->assertLessThan(0.0001,abs(1-$accuracy));
    }
}
