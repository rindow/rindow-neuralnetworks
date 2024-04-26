<?php
namespace Rindow\NeuralNetworks\Gradient;

use Countable;
use IteratorAggregate;
use ArrayAccess;

/**
 * @extends ArrayAccess<int,int>
 * @extends IteratorAggregate<int,int>
 */
interface ArrayShape extends ArrayAccess, Countable, IteratorAggregate
{}