<?php
namespace RindowTest\NeuralNetworks\Data\Dataset\NDArrayDatasetTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Data\Dataset\NDArrayDataset;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

class Test extends TestCase
{
    public function testNormal()
    {
        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);

        $inputs = $mo->array([
            [1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,12],
            [13,14,15],
        ]);
        $tests = $mo->array([
            [1],
            [2],
            [3],
            [4],
            [5],
        ]);
        $dataset = $nn->data->NDArrayDataset(
            $inputs,
            ['tests'=>$tests,'batch_size'=>2,'shuffle'=>false]
        );
        foreach ($dataset as $key => $value) {
            $datas[] = $value;
        }
        $this->assertCount(3,$datas);
        $this->assertEquals([
            [1,2,3],
            [4,5,6],
        ],$datas[0][0]->toArray());
        $this->assertEquals([
            [1],
            [2],
        ],$datas[0][1]->toArray());
        $this->assertEquals([
            [7,8,9],
            [10,11,12],
        ],$datas[1][0]->toArray());
        $this->assertEquals([
            [3],
            [4],
        ],$datas[1][1]->toArray());
        $this->assertEquals([
            [13,14,15],
        ],$datas[2][0]->toArray());
        $this->assertEquals([
            [5],
        ],$datas[2][1]->toArray());
    }
}
