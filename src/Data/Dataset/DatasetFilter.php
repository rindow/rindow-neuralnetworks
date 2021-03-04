<?php
namespace Rindow\NeuralNetworks\Data\Dataset;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

/**
 *
 */
interface DatasetFilter
{
    public function translate(
        iterable $inputs,
        iterable $tests=null,
        $options=null) : array;
}
