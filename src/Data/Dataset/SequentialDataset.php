<?php
namespace Rindow\NeuralNetworks\Data\Dataset;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use LogicException;
use Countable;
use IteratorAggregate;
use Traversable;
use function Rindow\Math\Matrix\R;

/**
 * @implements Dataset<NDArray>
 * @implements IteratorAggregate<int,array{NDArray|array<NDArray>,NDArray}>
 */
class SequentialDataset implements IteratorAggregate,Dataset
{
    protected object $mo;
    /** @var iterable<NDArray|array{NDArray,NDArray}> $iterable */
    protected iterable $iterable;
    protected ?NDArray $tests;
    protected int $batchSize;
    protected ?int $totalSize;
    protected bool $shuffle;
    /** @var DatasetFilter<NDArray> $filterGeneric */
    protected ?DatasetFilter $filterGeneric;
    protected mixed $inputsFilter;
    protected ?int $maxSize;
    protected bool $multiInputs=false;

    /**
     * @param iterable<NDArray|array{NDArray,NDArray}> $iterable
     * @param DatasetFilter<NDArray> $filter
     * @param DatasetFilter<NDArray> $inputs_filter
     */
    public function __construct(
        object $mo,
        iterable $iterable,
        ?int $batch_size=null,
        ?int $total_size=null,
        ?bool $shuffle=null,
        ?DatasetFilter $filter=null,
        ?DatasetFilter $inputs_filter=null,
    )
    {
        // defaults
        $batch_size = $batch_size ?? 32;
        $shuffle = $shuffle ?? true;
        $filter = $filter ?? null;

        $this->mo = $mo;
        $this->iterable = $iterable;
        $this->batchSize = $batch_size;
        $this->totalSize = $total_size;
        $this->shuffle = $shuffle;
        $this->setFilter($filter);
        $this->inputsFilter = $inputs_filter;
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
        if($this->totalSize===null) {
            throw new LogicException('total_size is not specified.');
        }
        return $this->totalSize;
    }

    public function count() : int
    {
        if($this->totalSize===null) {
            throw new LogicException('total_size is not specified.');
        }
        // It returns an estimate, but it is not an exact number.
        return (int)ceil($this->totalSize/$this->batchSize);
    }

    public function getIterator(): Traversable
    {
        $i = 0;
        $size = 0;
       foreach($this->iterable as $data) {
            if(is_array($data)) {
                [$inputs,$tests] = $data;
            } else {
                $inputs = $data;
                $tests = null;
            }
            if($this->inputsFilter!=null) {
                [$inputs,$tests] = $this->inputsFilter->translate($inputs,$tests);
            }
            $dataset = new NDArrayDataset(
                $this->mo,
                $inputs,
                $tests,
                $this->batchSize,
                $this->shuffle,
                $this->filterGeneric,
            );
            foreach($dataset as $return) {
                yield $i => $return;
                $i++;
                if($this->totalSize!==null) {
                    if(is_array($return[0])) {
                        $size += count($return[0][0]);
                    } else {
                        $size += count($return[0]);
                    }
                    if($size >= $this->totalSize) {
                        return;
                    }
                }
            }
        }
    }

}
