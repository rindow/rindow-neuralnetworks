<?php
declare(strict_types=1);
namespace Rindow\NeuralNetworks\Gradient\Core;

use InvalidArgumentException;
use LogicException;
use RuntimeException;
use ArrayAccess;
use Traversable;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Variable as VariableInterface;
use Rindow\NeuralNetworks\Gradient\ArrayShape as ArrayShapeInterface;
use Rindow\NeuralNetworks\Gradient\Scalar as ScalarInterface;


class Variable implements VariableInterface
{
    protected object $backend;
    protected bool $trainable;
    protected bool $undetermined;
    protected ?string $name;
    protected mixed $value;
    protected ?object $creator=null;
    protected int $generation=0;
    protected bool $unbackpropagatable;
    protected ?NDArray $mask;

    public function __construct(
        object $backend,
        mixed $value,
        string $name=null,
        bool $reference=null,
        bool $trainable=null,
        bool $undetermined=null,
        bool $unbackpropagatable=null,
        NDArray $mask=null,
    )
    {
        $this->backend = $backend;
        $this->undetermined = $undetermined ?? false;
        $this->name = $name;
        $this->trainable = $trainable ?? true;
        $this->unbackpropagatable = $unbackpropagatable ?? false;
        if(!$this->undetermined) {
            $this->assign($value, reference:$reference, mask:$mask);
        }
    }

    public function assign(
        mixed $value, bool $reference=null, NDArray $mask=null) : void
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
        } elseif($value instanceof ArrayShapeInterface) {
            $this->value = $value;
        } elseif($value instanceof ScalarInterface) {
            $this->value = $value;
        } else {
            throw new InvalidArgumentException('Invalid vaule type:'.gettype($value));
        }
        $this->mask = $mask;
        $this->undetermined = false;
    }

    public function mask() : NDArray
    {
        return $this->mask;
    }

    public function setMask(NDArray $mask) : void
    {
        $this->mask = $mask;
    }

    public function isTrainable() : bool
    {
        return $this->trainable;
    }

    public function isUndetermined() : bool
    {
        return $this->undetermined;
    }

    public function isbackpropagatable() : bool
    {
        return !$this->unbackpropagatable;
    }

    public function value() : mixed
    {
        if($this->undetermined) {
            throw new LogicException("Undetermined variable");
        }
        return $this->value;
    }

    public function name() : ?string
    {
        return $this->name;
    }

    public function setName(string $name) : string
    {
        $this->name = $name;
        return $name;
    }

    public function creator() : ?object
    {
        return $this->creator;
    }

    /**
     * @param object $creator
     */
    public function setCreator(object $creator) : void
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

    /**
     * @return array<int>
     */
    public function valueShape() : ?array
    {
        if($this->value===null) {
            return null;
        }
        $shape = $this->shape();
        array_shift($shape);
        return $shape;
    }

    public function reference() : object
    {
        return new VariableReference($this);
    }

    public function _clearValue() : void
    {
        $this->value = null;
    }

    public function dtype() : int
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

    /**
     * @return array<int>
     */
    public function shape() : array
    {
        $value = $this->value;
        if($value===null) {
            throw new LogicException('Variable has no value');
        }
        if(is_bool($value)) {
            return [];
        } elseif($value instanceof NDArray) {
            return $value->shape();
        }
        throw new RuntimeException('invalid type:'.(is_object($value)?get_class($value):gettype($value)));
    }

    public function ndim() : int
    {
        $value = $this->value;
        if($value===null) {
            throw new LogicException('Variable has no value');
        }
        if(is_bool($value)) {
            return 0;
        } elseif($value instanceof NDArray) {
            return $value->ndim();
        }
        throw new RuntimeException('invalid type:'.(is_object($value)?get_class($value):gettype($value)));
    }

    public function size() : int
    {
        $value = $this->value;
        if($value===null) {
            throw new LogicException('Variable has no value');
        }
        if(is_bool($value)) {
            return 0;
        } elseif($value instanceof NDArray) {
            return $value->size();
        }
        throw new RuntimeException('invalid type:'.(is_object($value)?get_class($value):gettype($value)));
    }

    /**
     * @return ArrayAccess<int,mixed>
     */
    public function buffer() : ArrayAccess
    {
        $value = $this->value;
        if($value===null) {
            throw new LogicException('Variable has no value');
        }
        if(is_bool($value)) {
            throw new LogicException('unsupported operation on boolean type');
        } elseif($value instanceof NDArray) {
            return $value->buffer();
        }
        throw new RuntimeException('invalid type:'.(is_object($value)?get_class($value):gettype($value)));
    }

    public function offset() : int
    {
        $value = $this->value;
        if($value===null) {
            throw new LogicException('Variable has no value');
        }
        if(is_bool($value)) {
            throw new LogicException('unsupported operation on boolean type');
        } elseif($value instanceof NDArray) {
            return $value->offset();
        }
        throw new RuntimeException('invalid type:'.(is_object($value)?get_class($value):gettype($value)));
    }

    /**
     * @param array<int> $shape
     */
    public function reshape(array $shape) : NDArray
    {
        $value = $this->value;
        if($value===null) {
            throw new LogicException('Variable has no value');
        }
        if(is_bool($value)) {
            throw new LogicException('unsupported operation on boolean type');
        } elseif($value instanceof NDArray) {
            return new self($this->backend,$value->reshape($shape), reference:true);
        }
        throw new RuntimeException('invalid type:'.(is_object($value)?get_class($value):gettype($value)));
    }

    public function toArray() : mixed
    {
        $value = $this->value;
        if($value===null) {
            throw new LogicException('Variable has no value');
        }
        if(is_bool($value)) {
            return $value;
        } elseif($value instanceof NDArray) {
            return $value->toArray();
        }
        throw new RuntimeException('invalid type:'.(is_object($value)?get_class($value):gettype($value)));
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
