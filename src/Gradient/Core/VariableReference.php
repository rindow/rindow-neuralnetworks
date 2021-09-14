<?php
namespace Rindow\NeuralNetworks\Gradient\Core;

class VariableReference
{
    protected $oid;
    protected $shape;
    protected $dtype;

    public function __construct(Variable $variable)
    {
        $this->oid = spl_object_hash($variable);
        $value = $variable->value();
        if($value!==null &&
           !($value instanceof UndeterminedNDArray &&
             $value->isNull())) {
            $this->shape = $value->shape();
            $this->dtype = $value->dtype();
        }
    }

    public function oid()
    {
        return $this->oid;
    }

    public function shape()
    {
        return $this->shape;
    }

    public function dtype()
    {
        return $this->dtype;
    }

    public function valueShape()
    {
        if($this->shape===null) {
            return null;
        }
        $shape = $this->shape;
        array_shift($shape);
        return $shape;
    }
}
