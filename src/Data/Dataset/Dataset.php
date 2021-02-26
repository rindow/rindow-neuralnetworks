<?php
namespace Rindow\NeuralNetworks\Data\Dataset;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use Countable;

/**
 *
 */
interface Dataset extends Countable // ,Traversable
{
    public function setFilter(DatasetFilter $filter) : void;
    public function batchSize() : int;
    public function datasetSize() : int;
}
