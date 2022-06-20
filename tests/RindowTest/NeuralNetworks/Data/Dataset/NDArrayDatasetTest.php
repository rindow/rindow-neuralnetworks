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
        $this->assertEquals(3,count($dataset));
        $this->assertEquals(5,$dataset->datasetSize());
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

    public function testNoTestData()
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
        $dataset = $nn->data->NDArrayDataset(
            $inputs,
            ['batch_size'=>2,'shuffle'=>false]
        );
        $this->assertEquals(3,count($dataset));
        $this->assertEquals(5,$dataset->datasetSize());
        foreach ($dataset as $key => $value) {
            $datas[] = $value;
        }
        $this->assertCount(3,$datas);
        $this->assertEquals([
            [1,2,3],
            [4,5,6],
        ],$datas[0][0]->toArray());
        $this->assertEquals(null,$datas[0][1]);
    }

    public function testMultiInput()
    {
        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);

        $a = $mo->array([
            [1,2,3],
            [4,5,6],
            [7,8,9],
        ]);
        $b = $mo->array([
            [11,12,13],
            [14,15,16],
            [17,18,19],
        ]);
        $inputs = [$a,$b];
        $tests = $mo->array([
            [1],
            [2],
            [3],
        ]);
        $dataset = $nn->data->NDArrayDataset(
            $inputs,
            ['tests'=>$tests,'batch_size'=>1,'shuffle'=>false]
        );
        $this->assertEquals(3,count($dataset));
        $this->assertEquals(3,$dataset->datasetSize());
        foreach ($dataset as $key => $value) {
            $datas[] = $value;
        }
        $this->assertCount(3,$datas);
        $this->assertTrue(is_array($datas[0]));
        [$inp,$tst] = $datas[0];
        $this->assertTrue(is_array($inp));
        $this->assertInstanceof(NDArray::class,$inp[0]);
        $this->assertInstanceof(NDArray::class,$inp[1]);
        $this->assertInstanceof(NDArray::class,$tst);

        //////////////
        $this->assertEquals([
            [1,2,3],
        ],$datas[0][0][0]->toArray());
        $this->assertEquals([
            [11,12,13],
        ],$datas[0][0][1]->toArray());
        $this->assertEquals([
            [1],
        ],$datas[0][1]->toArray());
        //////////////
        $this->assertEquals([
            [4,5,6],
        ],$datas[1][0][0]->toArray());
        $this->assertEquals([
            [14,15,16],
        ],$datas[1][0][1]->toArray());
        $this->assertEquals([
            [2],
        ],$datas[1][1]->toArray());
        //////////////
        $this->assertEquals([
            [7,8,9],
        ],$datas[2][0][0]->toArray());
        $this->assertEquals([
            [17,18,19],
        ],$datas[2][0][1]->toArray());
        $this->assertEquals([
            [3],
        ],$datas[2][1]->toArray());
    }

    public function testShuffle()
    {
        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);

        $a = $mo->array([
            [1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,12],
            [13,14,15],
        ]);
        $b = $mo->array([
            [101,102,103],
            [104,105,106],
            [107,108,109],
            [110,111,112],
            [113,114,115],
        ]);
        $inputs = [$a,$b];
        $tests = $mo->array([
            [1],
            [2],
            [3],
            [4],
            [5],
        ]);
        $dataset = $nn->data->NDArrayDataset(
            $inputs,
            ['tests'=>$tests,'batch_size'=>2,'shuffle'=>true]
        );
        $this->assertEquals(3,count($dataset));
        $this->assertEquals(5,$dataset->datasetSize());
        $serial = [];
        foreach ($dataset as $key => $value) {
            $datas[] = $value;
            [$inp,$t] = $value;
            [$da,$db] = $inp;
            $this->assertInstanceof(NDArray::class,$inp[0]);
            $this->assertInstanceof(NDArray::class,$inp[1]);
            $this->assertInstanceof(NDArray::class,$t);
            foreach($t as $i => $valueT) {
                $id = $valueT->toArray()[0];
                $this->assertFalse(isset($serial[$id]));
                $serial[$id] = true;
                switch($id) {
                    case 1: {
                        $this->assertEquals([1,2,3],$da[$i]->toArray());
                        $this->assertEquals([101,102,103],$db[$i]->toArray());
                        break;
                    }
                    case 2: {
                        $this->assertEquals([4,5,6],$da[$i]->toArray());
                        $this->assertEquals([104,105,106],$db[$i]->toArray());
                        break;
                    }
                    case 3: {
                        $this->assertEquals([7,8,9],$da[$i]->toArray());
                        $this->assertEquals([107,108,109],$db[$i]->toArray());
                        break;
                    }
                    case 4: {
                        $this->assertEquals([10,11,12],$da[$i]->toArray());
                        $this->assertEquals([110,111,112],$db[$i]->toArray());
                        break;
                    }
                    case 5: {
                        $this->assertEquals([13,14,15],$da[$i]->toArray());
                        $this->assertEquals([113,114,115],$db[$i]->toArray());
                        break;
                    }
                    default: {
                        $this->assertTrue(false);
                    }
                }
            }
        }
        $this->assertCount(3,$datas);
    }
}
