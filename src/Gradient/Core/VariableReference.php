<?php
namespace Rindow\NeuralNetworks\Gradient\Core;

use WeakReference;
use Interop\Polite\Math\Matrix\NDArray;

class VariableReference
{
    //protected $oid;
    protected object $ref;
    /** @var array<int> $shape */
    protected ?array $shape=null;
    protected int $dtype;

    public function __construct(Variable $variable)
    {
        //$this->oid = spl_object_id($variable);
        $this->ref = WeakReference::create($variable);
        $value = $variable->value();
        if($value instanceof NDArray) {
            $this->shape = $value->shape();
            $this->dtype = $value->dtype();
        }
    }

    //public function oid()
    //{
    //    return $this->oid;
    //}

    public function ref() : object
    {
        return $this->ref;
    }

    public function get() : ?object
    {
        return $this->ref->get();
    }

    /**
     * @param array<int> $shape
     */
    public function _setShape(array $shape) : void
    {
        $this->shape = $shape;
    }

    /**
     * @return array<int>
     */
    public function shape() : ?array
    {
        return $this->shape;
    }

    public function dtype() : int
    {
        return $this->dtype;
    }

    /**
     * @return array<int>
     */
    public function valueShape() : ?array
    {
        if($this->shape===null) {
            return null;
        }
        $shape = $this->shape;
        array_shift($shape);
        return $shape;
    }
}
