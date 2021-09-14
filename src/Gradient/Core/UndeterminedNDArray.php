<?php
namespace Rindow\NeuralNetworks\Gradient\Core;

use InvalidArgumentException;
use LogicException;
use ArrayAccess;
use Countable;
use Traversable;
use Interop\Polite\Math\Matrix\NDArray;

class UndeterminedNDArray implements NDArray
{
    protected $shape;
    protected $dtype;

    public function __construct(array $shape=null,$dtype=null)
    {
        $this->shape = $shape;
        $this->dtype = $dtype;
    }

    public function setShape(array $shape) : void
    {
        $this->shape = $shape;
    }

    public function isNull()
    {
        return ($this->shape===null);
    }

    public function shape() : array
    {
        if($this->shape===null) {
            throw new LogicException('Unsupported function');
        }
        return $this->shape;
    }

    public function dtype()
    {
        return $this->dtype;
    }

    public function ndim() : int
    {
        throw new LogicException('Unsupported function');
    }

    public function buffer() : ArrayAccess
    {
        throw new LogicException('Unsupported function');
    }

    public function offset() : int
    {
        throw new LogicException('Unsupported function');
    }

    public function size() : int
    {
        throw new LogicException('Unsupported function');
    }

    public function reshape(array $shape) : NDArray
    {
        throw new LogicException('Unsupported function');
    }

    public function toArray()
    {
        throw new LogicException('Unsupported function');
    }
    public function offsetExists($offset)
    {
        throw new LogicException('Unsupported function');
    }

    public function offsetGet($offset)
    {
        throw new LogicException('Unsupported function');
    }

    public function offsetSet($offset, $value)
    {
        throw new LogicException('Unsupported function');
    }

    public function offsetUnset($offset)
    {
        throw new LogicException('Unsupported function');
    }
}
