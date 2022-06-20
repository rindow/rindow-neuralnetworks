<?php
namespace Rindow\NeuralNetworks\Gradient;

use Countable;
use IteratorAggregate;
use Interop\Polite\Math\Matrix\NDArray;

interface Variable extends NDArray, Countable, IteratorAggregate
{}