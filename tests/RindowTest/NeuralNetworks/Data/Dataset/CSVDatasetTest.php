<?php
namespace RindowTest\NeuralNetworks\Data\Dataset\CSVDatasetTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Data\Dataset\CSVDataset;
use Rindow\NeuralNetworks\Data\Dataset\DatasetFilter;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;


class TestFilter implements DatasetFilter
{
    public function __construct($mo = null)
    {
        $this->mo = $mo;
    }

    public function translate(
        iterable $inputs, iterable $tests=null, $options=null) : array
    {
        $batchSize= count($inputs);
        $cols = count($inputs[0])-1;
        $inputsNDArray = $this->mo->la()->alloc([$batchSize,$cols]);
        $testsNDArray = $this->mo->la()->alloc([$batchSize,1]);
        foreach ($inputs as $i => $row) {
            $testsNDArray[$i][0] = (float)array_pop($row);
            for($j=0;$j<$cols;$j++) {
                $inputsNDArray[$i][$j] = (float)$row[$j];
            }
        }
        return [$inputsNDArray,$testsNDArray];
    }
}

class Test extends TestCase
{
    public function testNormal()
    {
        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);
        $filter = new TestFilter($mo);

        $dataset = $nn->data->CSVDataset(
            __DIR__.'/csv',
            pattern: '@.*\\.csv@',
            batch_size: 2,
            skip_header: true,
            filter: $filter,
            shuffle: false,
        );
        $datas = [];
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

        // epoch 2
        $datas = [];
        foreach ($dataset as $key => $value) {
            $datas[] = $value;
        }
        $this->assertCount(3,$datas);
    }
}
