<?php
declare(strict_types=1);
namespace Rindow\NeuralNetworks\Gradient\Core;

use InvalidArgumentException;
use LogicException;
use ArrayAccess;
use Traversable;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Variable as VariableInterface;


class Variable implements VariableInterface
{
    protected $backend;
    protected $trainable;
    protected $undetermined;
    protected $name;
    protected $value;
    protected $creator;
    protected $generation=0;

    public function __construct(
        object $backend,
        $value,
        string $name=null,
        bool $reference=null,
        bool $trainable=null,
        bool $undetermined=null,
    )
    {
        $this->backend = $backend;
        $this->undetermined = $undetermined;
        $this->name = $name;
        $this->trainable = $trainable ?? true;
        $undetermined = $undetermined ?? false;
        if(!$undetermined) {
            $this->assign($value, reference:$reference);
        }
    }

    public function assign($value, bool $reference=null) : void
    {
        $K = $this->backend;
        $reference = $reference ?? false;
        if($value instanceof VariableInterface) {
            $value = $value->value();
        }
        if($value instanceof NDArray) {
            if($reference) {
                $this->value = $value;
            } else {                                        // Copying NDArray before
                $this->value = $K->copy($K->array($value)); // translate from NDArray to NDArrayCL
            }                                               // if Backend is OpenCL.
        } elseif(is_bool($value)) {
            $this->value = $value;
        } elseif(is_array($value)||is_numeric($value)) {
            $this->value = $K->array($value);
        } else {
            throw new InvalidArgumentException('Invalid vaule type:'.gettype($value));
        }
        $this->undetermined = false;
    }

    public function isTrainable() : bool
    {
        return $this->trainable;
    }

    public function isUndetermined() : bool
    {
        return $this->undetermined;
    }

    public function value()
    {
        if($this->undetermined) {
            throw new LogicException("Undetermined variable");
        }
        return $this->value;
    }

    public function name()
    {
        return $this->name;
    }

    public function setName($name)
    {
        return $this->name = $name;
    }

    /**
    * @return Function $creator
    *   creater function
    */
    public function creator()
    {
        return $this->creator;
    }

    /**
    * @param Function $creator
    *   creater function
    * @return void
    */
    public function setCreator($creator) : void
    {
        if($this->trainable) {
            $this->creator = $creator;
        }
        $this->generation = $creator->generation() + 1;
    }

    public function generation() : int
    {
        return $this->generation;
    }

    public function valueShape()
    {
        if($this->value===null) {
            return null;
        }
        $shape = $this->shape();
        array_shift($shape);
        return $shape;
    }

    public function reference()
    {
        return new VariableReference($this);
    }

    public function _clearValue()
    {
        $this->value = null;
    }

    public function dtype()
    {
        $value = $this->value;
        if($value===null) {
            throw new LogicException('Variable has no value');
        }
        if($value instanceof NDArray) {
            return $value->dtype();
        }
        if(is_bool($value)) {
            return NDArray::bool;
        }
        throw new RuntimeException('invalid type:'.(is_object($value)?get_class($value):gettype($value)));
    }

    public function shape() : array
    {
        $value = $this->value;
        if($value===null) {
            throw new LogicException('Variable has no value');
        }
        if(is_bool($value)) {
            return [];
        }
        return $value->shape();
    }

    public function ndim() : int
    {
        $value = $this->value;
        if($value===null) {
            throw new LogicException('Variable has no value');
        }
        if(is_bool($value)) {
            return 0;
        }
        return $value->ndim();
    }

    public function size() : int
    {
        $value = $this->value;
        if($value===null) {
            throw new LogicException('Variable has no value');
        }
        if(is_bool($value)) {
            return 0;
        }
        return $value->size();
    }

    public function buffer() : ArrayAccess
    {
        $value = $this->value;
        if($value===null) {
            throw new LogicException('Variable has no value');
        }
        if(is_bool($value)) {
            throw new LogicException('unsupported operation on boolean type');
        }
        return $value->buffer();
    }

    public function offset() : int
    {
        $value = $this->value;
        if($value===null) {
            throw new LogicException('Variable has no value');
        }
        if(is_bool($value)) {
            throw new LogicException('unsupported operation on boolean type');
        }
        return $value->offset();
    }

    public function reshape(array $shape) : NDArray
    {
        $value = $this->value;
        if($value===null) {
            throw new LogicException('Variable has no value');
        }
        if(is_bool($value)) {
            throw new LogicException('unsupported operation on boolean type');
        }
        return new self($this->backend,$value->reshape($shape), reference:true);
    }

    public function toArray()
    {
        $value = $this->value;
        if($value===null) {
            throw new LogicException('Variable has no value');
        }
        if(is_bool($value)) {
            return $value;
        }
        return $value->toArray();
    }

    public function offsetExists( $offset ) : bool
    {
        $value = $this->value;
        if($value===null) {
            throw new LogicException('Variable has no value');
        }
        if(is_bool($value)) {
            throw new LogicException('unsupported operation on boolean type');
        }
        return $value->offsetExists($offset);
    }

    public function offsetGet( $offset ) : mixed
    {
        $value = $this->value;
        if($value===null) {
            throw new LogicException('Variable has no value');
        }
        if(is_bool($value)) {
            throw new LogicException('unsupported operation on boolean type');
        }
        return $value->offsetGet($offset);
    }

    public function offsetSet( $offset , $value ) : void
    {
        $value = $this->value;
        if($value===null) {
            throw new LogicException('Variable has no value');
        }
        if(is_bool($value)) {
            throw new LogicException('unsupported operation on boolean type');
        }
        $value->offsetSet($offset, $value);
    }

    public function offsetUnset( $offset ) : void
    {
        $value = $this->value;
        if($value===null) {
            throw new LogicException('Variable has no value');
        }
        if(is_bool($value)) {
            throw new LogicException('unsupported operation on boolean type');
        }
        $value->offsetUnset($offset);
    }

    public function count() : int
    {
        $value = $this->value;
        if($value===null) {
            throw new LogicException('Variable has no value');
        }
        if(is_bool($value)) {
            throw new LogicException('unsupported operation on boolean type');
        }
        return $value->count();
    }

    public function  getIterator() :  Traversable
    {
        $value = $this->value;
        if($value===null) {
            throw new LogicException('Variable has no value');
        }
        if(is_bool($value)) {
            throw new LogicException('unsupported operation on boolean type');
        }
        return $value->getIterator();
    }

    public function __clone()
    {
        if(is_object($this->value)) {
            $this->value = clone $this->value;
        }
    }
}
