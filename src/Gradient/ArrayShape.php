<?php
namespace Rindow\NeuralNetworks\Gradient;

use Countable;
use IteratorAggregate;
use ArrayAccess;

interface ArrayShape extends ArrayAccess, Countable, IteratorAggregate
{}