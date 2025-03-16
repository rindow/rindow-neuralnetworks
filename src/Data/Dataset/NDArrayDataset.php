<?php
namespace Rindow\NeuralNetworks\Data\Dataset;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use Countable;
use IteratorAggregate;
use Traversable;
use function Rindow\Math\Matrix\R;

/**
 * @implements Dataset<NDArray>
 * @implements IteratorAggregate<int,array{NDArray|array<NDArray>,NDArray}>
 */
class NDArrayDataset implements IteratorAggregate,Dataset
{
    protected object $mo;
    /** @var array<NDArray> $inputs */
    protected array $inputs;
    protected ?NDArray $tests;
    protected int $batchSize;
    protected bool $shuffle;
    /** @var DatasetFilter<NDArray> $filterGeneric */
    protected ?DatasetFilter $filterGeneric;
    protected bool $multiInputs=false;

    /**
     * @param array<NDArray>|NDArray $inputs
     * @param DatasetFilter<NDArray> $filter
     */
    public function __construct(
        object $mo,
        array|NDArray $inputs,
        ?NDArray $tests=null,
        ?int $batch_size=null,
        ?bool $shuffle=null,
        ?DatasetFilter $filter=null,
    )
    {
        // defaults
        $tests = $tests ?? null;
        $batch_size = $batch_size ?? 32;
        $shuffle = $shuffle ?? true;
        $filter = $filter ?? null;

        $this->mo = $mo;
        if(is_array($inputs)) {
            $this->multiInputs = true;
        } else {
            $inputs = [$inputs];
        }
        $inputCount = null;
        foreach ($inputs as $value) {
            if(!($value instanceof NDArray)) {
                throw new InvalidArgumentException('inputs must be NDArray or NDArray list');
            }
            if($inputCount!==null) {
                if($inputCount!=count($value)) {
                    throw new InvalidArgumentException('All data contained in inputs must be the same length');
                }
            } else {
                $inputCount = count($value);
            }
        }
        if($inputCount===null) {
            throw new InvalidArgumentException('inputs is empty');
        }
        $this->inputs = $inputs;
        if($tests!==null) {
            if($inputCount!=count($tests)) {
                throw new InvalidArgumentException(
                    "Unmatch data size of inputs and tests:".$inputCount.",".count($tests));
            }
        }
        $this->tests = $tests;
        $this->batchSize = $batch_size;
        $this->shuffle = $shuffle;
        $this->setFilter($filter);
    }

    public function filter() : ?DatasetFilter
    {
        return $this->filterGeneric;
    }

    public function setFilter(?DatasetFilter $filter) : void
    {
        $this->filterGeneric = $filter;
    }

    public function batchSize() : int
    {
        return $this->batchSize;
    }

    public function datasetSize() : int
    {
        return count($this->inputs[0]);
    }

    public function count() : int
    {
        return (int)ceil(count($this->inputs[0])/$this->batchSize);
    }

    public function  getIterator() : Traversable
    {
        $la = $this->mo->la();
        if(count($this->inputs[0])==0)
            return [];
        $count = count($this->inputs[0]);
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
            $end = ($idx+1)*$batchSize;
            if($end>$count) {
                $end = $count;
            }
            $inputs = [];
            foreach ($this->inputs as $value) {
                $inputs[] = $value[R($start,$end)];
            }
            $tests = null;
            if($this->tests) {
                $tests = $this->tests[R($start,$end)];
            }
            if($this->filter()) {
                if($this->multiInputs) {
                    [$inputs,$tests] = $this->filter()->translate($inputs,$tests);
                } else {
                    [$inputs,$tests] = $this->filter()->translate($inputs[0],$tests);
                    $inputs = [$inputs];
                }
            }
            if($this->shuffle) {
                $size = $end-$start;
                if($size>1) {
                    $choiceItem = $la->randomSequence($size);
                } else {
                    $choiceItem = $la->alloc([1],NDArray::int32);
                    $la->zeros($choiceItem);
                }
                $orgInputs = $inputs;
                $inputs = [];
                foreach ($orgInputs as $key => $value) {
                    $inputs[] = $la->gatherb($value,$choiceItem);
                }
                unset($orgInputs);
                if($tests!==null) {
                    $tests  = $la->gatherb($tests,$choiceItem);
                }
            }
            if(!$this->multiInputs) {
                $inputs = $inputs[0];
            }
            yield $i => [$inputs,$tests];
        }
    }
}
