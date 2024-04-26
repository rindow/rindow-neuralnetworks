<?php
namespace Rindow\NeuralNetworks\Support\HDA;

use ArrayAccess;
use IteratorAggregate;

/**
 * Hierarchical Data Access
 * @extends ArrayAccess<mixed,mixed>
 * @extends IteratorAggregate<mixed,mixed>
 */
interface HDA extends ArrayAccess,IteratorAggregate
{
}
