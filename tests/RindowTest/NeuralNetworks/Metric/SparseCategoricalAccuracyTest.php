<?php
namespace RindowTest\NeuralNetworks\Metric\SparseCategoricalAccuracyTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Plot\Plot;

class SparseCategoricalAccuracyTest extends TestCase
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
        $metric = $nn->metrics->SparseCategoricalAccuracy();

        $x = $K->array($mo->array([
            [0.00000, 0.00000 , 1.00000],
            [0.99998, 0.00001 , 0.00001],
        ]));
        $t = $K->array($mo->array([2, 0],NDArray::int32));
        
        $copyx = $K->copy($x);
        $copyt = $K->copy($t);

        $metric->update($t,$x);
        $accuracy = $metric->result();
        $this->assertLessThan(0.0001,abs(1-$accuracy));
        $this->assertEquals($copyx->toArray(),$x->toArray());
        $this->assertEquals($copyt->toArray(),$t->toArray());

        $x = $K->array($mo->array([
            [0.00000, 0.00000 , 1.00000],
            [0.00001, 0.99998 , 0.00001],
        ]));
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
        $metric = $nn->metrics->SparseCategoricalAccuracy();

        /////////////////////////////////////
        $x = $K->zeros([2,3,4]);
        $t = $K->zeros([2,3],dtype:NDArray::int32);

        $metric->reset();
        $metric->update($t,$x);
        $accuracy = $metric->result();
        $this->assertLessThan(0.0001,abs(1-$accuracy));

        /////////////////////////////////////
        $x = $K->array($mo->array([
            [[0.00000, 0.00000 , 1.00000],
             [0.99998, 0.00001 , 0.00001]],
            [[0.00000, 0.00000 , 1.00000],
             [0.99998, 0.00001 , 0.00001]],
        ]));
        $t = $K->array($mo->array([
            [2, 0],
            [2, 0],
        ],NDArray::int32));

        $metric->reset();
        $metric->update($t,$x);
        $accuracy = $metric->result();
        $this->assertLessThan(0.0001,abs(1-$accuracy));
    }
}
