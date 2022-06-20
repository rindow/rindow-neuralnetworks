<?php
namespace RindowTest\NeuralNetworks\Data\Image\ImageFilterTest;

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
            [[[1],[2],[3]],[[4],[5],[6]],[[7],[8],[9]]],
            [[[10],[11],[12]],[[13],[14],[15]],[[16],[17],[18]]],
            [[[1],[2],[3]],[[4],[5],[6]],[[7],[8],[9]]],
        ]);
        $tests = $mo->array([
            [1],
            [2],
            [3],
        ]);
        $filter = $nn->data->ImageFilter(
            height_shift: 0.8,
            width_shift: 2,
            vertical_flip: true,
            horizontal_flip: true,
        );
        $dataset = $nn->data->NDArrayDataset(
            $inputs,
            tests: $tests,
            batch_size: 2,
            shuffle: false,
            filter: $filter,
        );
        foreach ($dataset as $key => $value) {
            $datas[] = $value;
        }
        $this->assertCount(2,$datas);
        //echo $mo->toString($datas[0][0]->reshape([2,3,3]),null,true)."\n";
        //echo $mo->toString($datas[1][0]->reshape([1,3,3]),null,true)."\n";
        $this->assertEquals([2,3,3,1],$datas[0][0]->shape());
        $this->assertEquals([
            [1],
            [2],
        ],$datas[0][1]->toArray());
        $this->assertEquals([1,3,3,1],$datas[1][0]->shape());
        $this->assertEquals([
            [3],
        ],$datas[1][1]->toArray());
    }

    public function testImageDataGenerator()
    {
        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);

        $inputs = $mo->array([
            [[[1],[2],[3]],[[4],[5],[6]],[[7],[8],[9]]],
            [[[10],[11],[12]],[[13],[14],[15]],[[16],[17],[18]]],
            [[[1],[2],[3]],[[4],[5],[6]],[[7],[8],[9]]],
        ]);
        $tests = $mo->array([
            [1],
            [2],
            [3],
        ]);
        $dataset = $nn->data->ImageDataGenerator(
            $inputs,
            tests: $tests,
            batch_size: 2,
            height_shift: 0.8,
            width_shift: 2,
            vertical_flip: true,
            horizontal_flip: true,
            shuffle: false,
        );
        foreach ($dataset as $key => $value) {
            $datas[] = $value;
        }
        $this->assertCount(2,$datas);
        //echo $mo->toString($datas[0][0]->reshape([2,3,3]),null,true)."\n";
        //echo $mo->toString($datas[1][0]->reshape([1,3,3]),null,true)."\n";
        $this->assertEquals([2,3,3,1],$datas[0][0]->shape());
        $this->assertEquals([
            [1],
            [2],
        ],$datas[0][1]->toArray());
        $this->assertEquals([1,3,3,1],$datas[1][0]->shape());
        $this->assertEquals([
            [3],
        ],$datas[1][1]->toArray());
    }
}
