<?php
namespace Rindow\NeuralNetworks\Data\Dataset;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

/**
 * @template T
 */
interface DatasetFilter
{
    /**
     * @param iterable<T> $inputs
     * @param iterable<T> $tests
     * @param array<mixed,mixed> $options
     * @return array{T,T}  {inputsArray,testsArray}
     */
    public function translate(
        iterable $inputs,
        iterable $tests=null,
        array $options=null) : array;
}
