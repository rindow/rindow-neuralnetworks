<?php
namespace Rindow\NeuralNetworks\Gradient\Core;

use InvalidArgumentException;
use LogicException;
use Traversable;
use Rindow\NeuralNetworks\Gradient\ArrayShape as ArrayShapeInterface;

class ArrayShape implements ArrayShapeInterface
{
    protected $shape;

    public function __construct(array $shape)
    {
        $this->shape = $shape;
    }

    public function offsetExists( $offset ) : bool
    {
        if(!is_int($offset)) {
            throw new LogicException('offset must be int');
        }
        if(!isset($this->shape[$offset])) {
            return false;
        }
        return true;
    }

    public function offsetGet( $offset ) : mixed
    {
        if(!$this->offsetExists($offset)) {
            var_dump(array_keys($this->shape));
            throw new LogicException('no found the offset: '.$offset);
        }
        return $this->shape[$offset];
    }

    public function offsetSet( $offset , $value ) : void
    {
        throw new LogicException('unsupported operation on boolean type');
    }

    public function offsetUnset( $offset ) : void
    {
        throw new LogicException('unsupported operation on boolean type');
    }

    public function count() : int
    {
        return count($this->shape);
    }

    public function  getIterator() :  Traversable
    {
        foreach($this->shape as $i => $v) {
            yield $i => $v;
        }
    }
}