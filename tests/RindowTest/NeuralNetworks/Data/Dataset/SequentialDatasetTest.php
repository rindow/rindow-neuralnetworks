<?php
namespace RindowTest\NeuralNetworks\Data\Dataset\SequentialDatasetTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Data\Dataset\DatasetFilter;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use ArrayIterator;

class SequentialDatasetTest extends TestCase
{
    public function testNormal()
    {
        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);

        $inputs = [
            [$mo->array([[1,1,1],[1,1,1],[1,1,1],[1,1,1]]),$mo->array([[1],[1],[1],[1]])],
            [$mo->array([[2,2,2],[2,2,2],[2,2,2],[2,2,2]]),$mo->array([[2],[2],[2],[2]])],
            [$mo->array([[3,3,3],[3,3,3],[3,3,3],[3,3,3]]),$mo->array([[3],[3],[3],[3]])],
        ];
        $inputs = new ArrayIterator($inputs);
        $dataset = $nn->data->SequentialDataset(
            $inputs,
            batch_size: 2, shuffle: false,
            total_size: 4*3,
        );
        $this->assertEquals(6,count($dataset));
        $this->assertEquals(12,$dataset->datasetSize());
        $idx = 0;
        foreach ($dataset as $key => $values) {
            [$value,$test] = $values;
            $this->assertEquals($idx,$key);
            $v = intdiv($idx,4/2)+1;
            $this->assertEquals([[$v,$v,$v],[$v,$v,$v]],$value->toArray());
            $this->assertEquals([[$v],[$v]],$test->toArray());
            $idx++;
        }

        $this->assertEquals(6,$idx);
    }

    public function testFilter()
    {
        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);

        $inputs = [
            [$mo->array([[1,1,1],[1,1,1],[1,1,1],[1,1,1]]),$mo->array([[1],[1],[1],[1]])],
            [$mo->array([[2,2,2],[2,2,2],[2,2,2],[2,2,2]]),$mo->array([[2],[2],[2],[2]])],
            [$mo->array([[3,3,3],[3,3,3],[3,3,3],[3,3,3]]),$mo->array([[3],[3],[3],[3]])],
        ];
        $inputs = new ArrayIterator($inputs);

        $filter = new class ($mo) implements DatasetFilter
        {
            protected object $mo;
            protected float $alpha = 1;
            public function __construct(object $mo) {
                $this->mo = $mo;
            }
            public function translate(
                iterable $inputs,
                ?iterable $tests=null,
                ?array $options=null) : array
            {
                $la = $this->mo->la();
                $this->alpha += 1;
                return [
                    $la->scal($this->alpha, $la->copy($inputs)),
                    $la->scal($this->alpha, $la->copy($tests)),
                ];
            }
        };
        $dataset = $nn->data->SequentialDataset(
            $inputs,
            batch_size: 2, shuffle: false,
            total_size: 4*3,
            filter: $filter,
        );
        $this->assertEquals(6,count($dataset));
        $this->assertEquals(12,$dataset->datasetSize());
        $idx = 0;
        foreach ($dataset as $key => $values) {
            [$value,$test] = $values;
            $this->assertEquals($idx,$key);
            $v = intdiv($idx,4/2)+1;
            $v *= ($idx+2);
            $this->assertEquals([[$v,$v,$v],[$v,$v,$v]],$value->toArray());
            $this->assertEquals([[$v],[$v]],$test->toArray());
            $idx++;
        }

        $this->assertEquals(6,$idx);
    }

    public function testBatchBoundary()
    {
        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);

        $inputs = [
            [$mo->array([[1,1,1],[1,1,1],[1,1,1],[1,1,1]]),$mo->array([[1],[1],[1],[1]])],
            [$mo->array([[2,2,2],[2,2,2],[2,2,2],[2,2,2]]),$mo->array([[2],[2],[2],[2]])],
            [$mo->array([[3,3,3],[3,3,3],[3,3,3],[3,3,3]]),$mo->array([[3],[3],[3],[3]])],
        ];
        $inputs = new ArrayIterator($inputs);

        $dataset = $nn->data->SequentialDataset(
            $inputs,
            batch_size: 3, shuffle: false,
            total_size: 4*3,
        );
        $this->assertEquals(4,count($dataset));
        $this->assertEquals(12,$dataset->datasetSize());

        $idx = 0;
        foreach ($dataset as $key => $values) {
            [$value,$test] = $values;
            $this->assertEquals($idx,$key);
            if($idx%2==0) {
                $this->assertEquals([3,3],$value->shape());
                $this->assertEquals([3,1],$test->shape());
            } else {
                $this->assertEquals([1,3],$value->shape());
                $this->assertEquals([1,1],$test->shape());
            }
            $idx++;
        }

        $this->assertEquals(6,$idx);
    }

    public function testInputsFilter()
    {
        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);

        $inputs = [
            [$mo->array([[1,1,1],[1,1,1],[1,1,1],[1,1,1]]),$mo->array([[1],[1],[1],[1]])],
            [$mo->array([[2,2,2],[2,2,2],[2,2,2],[2,2,2]]),$mo->array([[2],[2],[2],[2]])],
            [$mo->array([[3,3,3],[3,3,3],[3,3,3],[3,3,3]]),$mo->array([[3],[3],[3],[3]])],
        ];
        $inputs = new ArrayIterator($inputs);
        $inputsFilter = new class ($mo) implements DatasetFilter
        {
            protected object $mo;
            protected float $alpha = 1;
            public function __construct(object $mo) {
                $this->mo = $mo;
            }
            public function translate(
                iterable $inputs,
                ?iterable $tests=null,
                ?array $options=null) : array
            {
                $la = $this->mo->la();
                $this->alpha += 1;
                return [
                    $la->scal($this->alpha, $la->copy($inputs)),
                    $la->scal($this->alpha, $la->copy($tests)),
                ];
            }
        };

        $dataset = $nn->data->SequentialDataset(
            $inputs,
            batch_size: 2, shuffle: false,
            total_size: 4*3,
            inputs_filter: $inputsFilter,
        );
        $this->assertEquals(6,count($dataset));
        $this->assertEquals(12,$dataset->datasetSize());

        $idx = 0;
        foreach ($dataset as $key => $values) {
            [$value,$test] = $values;
            $this->assertEquals($idx,$key);
            $v = intdiv($idx,4/2)+1;
            $v *= (intdiv($idx,2)+2);
            $this->assertEquals([[$v,$v,$v],[$v,$v,$v]],$value->toArray());
            $this->assertEquals([[$v],[$v]],$test->toArray());
            $idx++;
        }

        $this->assertEquals(6,$idx);
    }

    public function testMaxSize()
    {
        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);

        $inputs = [
            [$mo->array([[1,1,1],[1,1,1],[1,1,1],[1,1,1]]),$mo->array([[1],[1],[1],[1]])],
            [$mo->array([[2,2,2],[2,2,2],[2,2,2],[2,2,2]]),$mo->array([[2],[2],[2],[2]])],
            [$mo->array([[3,3,3],[3,3,3],[3,3,3],[3,3,3]]),$mo->array([[3],[3],[3],[3]])],
        ];
        $inputs = new ArrayIterator($inputs);
        $dataset = $nn->data->SequentialDataset(
            $inputs,
            batch_size: 2, shuffle: false,
            total_size: 5,
        );
        $this->assertEquals(3,count($dataset));
        $this->assertEquals(5,$dataset->datasetSize());

        $idx = 0;
        foreach ($dataset as $key => $values) {
            [$value,$test] = $values;
            $this->assertEquals($idx,$key);
            $idx++;
        }

        $this->assertEquals(3,$idx);
    }
}
