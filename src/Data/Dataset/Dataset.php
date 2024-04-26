<?php
namespace Rindow\NeuralNetworks\Data\Dataset;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use Countable;
use Traversable;

/**
 * @template T
 * @extends Traversable<int,array{NDArray|array<NDArray>,NDArray}>
 */
interface Dataset extends Countable,Traversable
{
    /**
     * @return DatasetFilter<T>
     */
    public function filter() : ?DatasetFilter;
    /**
     * @param DatasetFilter<T> $filter
     */
    public function setFilter(?DatasetFilter $filter) : void;
    public function batchSize() : int;
    public function datasetSize() : int;
}
