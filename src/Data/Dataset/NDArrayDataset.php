<?php
namespace Rindow\NeuralNetworks\Data\Dataset;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;
use InvalidArgumentException;
use Countable;
use IteratorAggregate;

class NDArrayDataset implements Countable,IteratorAggregate,Dataset
{
    use GenericUtils;
    protected $mo;
    protected $inputs;
    protected $tests;
    protected $batchSize;
    protected $shuffle;
    protected $filter;

    public function __construct(
        $mo,
        NDArray $inputs,
        array $options=null,
        array &$leftargs=null
        )
    {
        extract($this->extractArgs([
            'tests'=>null,
            'batch_size'=>32,
            'shuffle'=>true,
            'filter'=>null,
        ],$options,$leftargs));
        $this->mo = $mo;
        $this->inputs = $inputs;
        if($tests!==null) {
            if(count($inputs)!=count($tests)) {
                throw new InvalidArgumentException(
                    "Unmatch data size of inputs and tests:".count($inputs).",".count($tests));
            }
        }
        $this->tests = $tests;
        $this->batchSize = $batch_size;
        $this->shuffle = $shuffle;
        $this->filter = $filter;
    }

    public function setFilter(DatasetFilter $filter) : void
    {
        $this->filter = $filter;
    }

    public function batchSize() : int
    {
        return $this->batchSize;
    }

    public function datasetSize() : int
    {
        return count($this->inputs);
    }

    public function count()
    {
        return (int)ceil(count($this->inputs)/$this->batchSize);
    }

    public function  getIterator()
    {
        $la = $this->mo->la();
        if(count($this->inputs)==0)
            return [];
        $count = count($this->inputs);
        $batchSize = $this->batchSize;
        $steps = (int)ceil($count/$batchSize);
        if($this->shuffle&&$steps>1) {
            $choice = $la->randomSequence($steps);
        } else {
            $choice = $this->mo->arange($steps);
        }
        for($i=0;$i<$steps;$i++) {
            $idx = $choice[$i];
            $start = $idx*$batchSize;
            $end = ($idx+1)*$batchSize-1;
            if($end>=$count) {
                $end = $count-1;
            }
            $inputs = $this->inputs[[$start,$end]];
            $tests = null;
            if($this->tests) {
                $tests = $this->tests[[$start,$end]];
            }
            if($this->filter) {
                [$inputs,$tests] = $this->filter->translate($inputs,$tests);
            }
            if($this->shuffle) {
                $size = $end-$start+1;
                if($size>1) {
                    $choiceItem = $la->randomSequence($size);
                } else {
                    $choiceItem = $la->alloc([1],NDArray::int32);
                    $la->zeros($choiceItem);
                }
                $inputs = $la->gather($inputs,$choiceItem);
                if($tests!==null) {
                    $tests  = $la->gather($tests,$choiceItem);
                }
            }
            yield $i => [$inputs,$tests];
        }
    }
}
