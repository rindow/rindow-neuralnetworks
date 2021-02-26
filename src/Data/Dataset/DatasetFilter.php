<?php
namespace Rindow\NeuralNetworks\Data\Dataset;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

/**
 *
 */
interface DatasetFilter
{
    public function translate($inputs, $tests=null) : array;
}
