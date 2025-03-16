<?php
namespace Rindow\NeuralNetworks\Gradient\Core;

use ArrayAccess;
use Traversable;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\MaskedNDArray as MaskedNDArrayInterface;
use Rindow\Math\Matrix\Drivers\Service;


class MaskedNDArray implements MaskedNDArrayInterface
{
    protected NDArray $value;
    protected NDArray $mask;

    public function __construct(
        NDArray $value,
        NDArray $mask
    )
    {
        $this->value = $value;
        $this->mask = $mask;
    }

    public function value() : NDArray
    {
        return $this->value;
    }

    public function mask() : NDArray
    {
        return $this->mask;
    }

    public function dtype() : int
    {
        return $this->value->dtype();
    }

    /**
     * @return array<int>
     */
    public function shape() : array
    {
        return $this->value->shape();
    }

    public function ndim() : int
    {
        return $this->value->ndim();
    }

    public function size() : int
    {
        return $this->value->size();
    }

    /**
     * @return ArrayAccess<int,mixed>
     */
    public function buffer() : ArrayAccess
    {
        return $this->value->buffer();
    }

    public function offset() : int
    {
        return $this->value->offset();
    }

    /**
     * @param array<int> $shape
     */
    public function reshape(array $shape) : NDArray
    {
        return $this->value->reshape($shape);
    }

    public function toArray() : mixed
    {
        return $this->value->toArray();
    }

    public function offsetExists( $offset ) : bool
    {
        return $this->value->offsetExists($offset);
    }

    public function offsetGet( $offset ) : mixed
    {
        return $this->value->offsetGet($offset);
    }

    public function offsetSet( $offset , $value ) : void
    {
        $this->value->offsetSet($offset, $value);
    }

    public function offsetUnset( $offset ) : void
    {
        $this->value->offsetUnset($offset);
    }

    public function count() : int
    {
        return $this->value->count();
    }

    public function  getIterator() :  Traversable
    {
        return $this->value->getIterator();
    }

    public function service() : Service
    {
        return $this->value->service();
    }

    public function __clone()
    {
        $this->value = clone $this->value;
        $this->mask = clone $this->mask;
    }
}
