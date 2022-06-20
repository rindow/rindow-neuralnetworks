<?php
namespace Rindow\NeuralNetworks\Gradient\Core;

use InvalidArgumentException;
use LogicException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class Variable
{
    use GenericUtils;
    protected $backend;
    protected $value;
    protected $grad;
    protected $creator;
    protected $generation=0;
    protected $name;

    public function __construct($backend, NDArray $value, $options=null)
    {
        extract($this->extractArgs([
            'name'=>null,
        ],$options));
        $this->backend = $backend;
        $this->value = $value;
        $this->name = $name;
    }

    public function value()
    {
        return $this->value;
    }

    public function grad()
    {
        return $this->grad;
    }

    public function setGrad(NDArray $grad)
    {
        $this->grad = $grad;
    }

    public function name()
    {
        return $this->name;
    }

    public function setName($name)
    {
        return $this->name = $name;
    }

    public function dtype()
    {
        if($this->value===null) {
            throw new LogicException('Variable has no value');
        }
        return $this->value->dtype();
    }

    public function shape()
    {
        if($this->value===null) {
            throw new LogicException('Variable has no value');
        }
        return $this->value->shape();
    }

    public function ndim()
    {
        if($this->value===null) {
            throw new LogicException('Variable has no value');
        }
        return $this->value->ndim();
    }

    public function size()
    {
        if($this->value===null) {
            throw new LogicException('Variable has no value');
        }
        return $this->value->size();
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
        $this->creator = $creator;
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
}
