<?php
namespace Rindow\NeuralNetworks\Gradient;

use Countable;
use IteratorAggregate;
use Interop\Polite\Math\Matrix\NDArray;

/**
 * @extends IteratorAggregate<mixed>
 */
interface Variable extends NDArray, Countable, IteratorAggregate
{
    public function name() : ?string;
    public function isTrainable() : bool;
    public function isUndetermined() : bool;
    public function isbackpropagatable() : bool;
    public function _assign(mixed $value, ?bool $reference=null, ?NDArray $mask=null) : void;
    public function update(mixed $value) : void;
    public function value() : mixed;
    public function creator() : ?object;
    public function setCreator(object $creator) : void;
    public function generation() : int;
    public function reference() : object;
}