<?php
declare(strict_types=1);
namespace Rindow\NeuralNetworks\Gradient\Core;

use InvalidArgumentException;
use stdClass;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Scalar as ScalarInterface;
use Rindow\NeuralNetworks\Gradient\ArrayShape as ArrayShapeInterface;
use Rindow\NeuralNetworks\Gradient\Core\Scalar;

abstract class AbstractFunction
{
    use GradientUtils;
    protected ?array $unbackpropagatables = null;

    /**
    *  @param array<NDArray>  $inputs
    *       inputs
    *  @return array<NDArray>
    *       outputs
    */
    abstract protected function call(array $inputs) : array;

    /**
    *  @param array<NDArray>  $dOutputs
    *       difference outputs
    *  @return array<NDArray>
    *       difference inputs
    */
    abstract protected function differentiate(array $dOutputs) : array;

    protected $backend;

    /**
    *  @var array<Variable>   inputs
    */
    protected $inputsVariables;

    /**
    *  @var array<Variable>   outputs
    */
    protected $outputsVariables;

    /**
    *  @var int   generation
    */
    protected $generation;

    /**
    *  @var int   numOfInputs
    */
    protected $numOfInputs = 1;

    /**
    *  @var int   numOfOutputs
    */
    protected $numOfOutputs = 1;

    protected $container;

    public function __construct(object $backend)
    {
        $this->backend = $backend;
        $this->container = new stdClass();
    }

    /**
    *  @return array<Variable>
    *       outputs
    */
    public function inputs() : array
    {
        return $this->inputsVariables;
    }

    /**
    *  @return Dict<Variable>
    *       options
    */
    public function options() : array
    {
        return [];
    }

    /**
    *  @return array<Variable>
    *       outputs
    */
    public function outputs() : array
    {
        return $this->outputsVariables;
    }

    /**
    *  @return int
    */
    public function generation() : int
    {
        return $this->generation;
    }

    /**
     * Call from SessionFunc in compiled graph
     */
    public function _rawCall(array $inputs,array $options)
    {
        return $this->call($inputs);
    }

    public function className() : string
    {
        return get_class($this);
    }

    protected function preprocess(array $inputs) : array
    {
        return $inputs;
    }

    protected function toScalar($value,$argIdx) : mixed
    {
        $K = $this->backend;
        if($value instanceof ScalarInterface) {
            $value = $value->value();
        } elseif($value instanceof NDArray) {
            if($value->ndim()!=0) {
                throw new InvalidArgumentException("arg #$argIdx must not be scalar.");
            }
            $value = $K->scalar($value);
        } else {
            if(is_object($value)) {
                $type = get_class($value);
            } else {
                $type = gettype($value);
            }
            throw new InvalidArgumentException("arg #$argIdx is invalid data type.: $type");
        }
        return $value;
    }

    protected function extractShapeArgment(mixed $shape) : array
    {
        if($shape instanceof Variable) {
            if($shape->value() instanceof ArrayShapeInterface) {
                return [$shape];
            }
            throw new InvalidArgumentException('shape must be array or ShapeArray');
        }
        if($shape instanceof ArrayShapeInterface) {
            return [$shape];
        }
        if(!is_array($shape)) {
            throw new InvalidArgumentException('shape must be array or ShapeArray');
        }
        if(count($shape)==0) {
            return [new ArrayShape([])];
        }
        ksort($shape);
        if(array_reduce($shape,fn($t,$v)=>$t && is_numeric($v),true)) { // numeric all
            return [new ArrayShape($shape)];
        }
        $inputs = array_map(fn($v)=>is_numeric($v)?(new Scalar($v)):$v,$shape);
        return $inputs;
    }

    protected function translateToShape(array $inputs) : array
    {
        if($inputs[0] instanceof ArrayShapeInterface) {
            $inputs = $inputs[0];
        }
        $shape = [];
        foreach ($inputs as $value) {
            if($value instanceof ScalarInterface) {
                $value = $value->value();
            }
            if(!is_int($value)) {
                throw new InvalidArgumentException('shape must contains integer values.');
            }
            $shape[] = $value;
        }
        return $shape;
    }

    /**
    *  @param array<Variable>  $inputs
    *       inputs
    *  @return array<Variable>
    *       outputs
    */
    public function __invoke(...$inputs)
    {
        if(count($inputs)!=$this->numOfInputs) {
            throw new InvalidArgumentException($this->numOfInputs.' arguments are required.');
        }
        $inputs = $this->preprocess($inputs);
        [$inputs,$rawInputs]     = $this->packAndUnpackVariables($this->backend,$inputs);
        if(GraphFunction::$mode==GraphFunction::EXECUTING) {
            $outputs = $this->call($inputs);
            if(count($outputs)==1) {
                return $outputs[0];
            }
            return $outputs;
        }
        if(GradientTape::$autoBackProp) {
            $this->inputsVariables = $inputs;
        }
        $this->unbackpropagatables = null;
        $outValues = $this->call($rawInputs);
        $outputs = $this->postGradientProcess($this->backend,$inputs,
                                                $outValues,$this->unbackpropagatables);

        if(count($outputs)==1) {
            return $outputs[0];
        }
        return $outputs;
    }

    /**
    *  @param array<NDArray>  $inputs
    *       inputs
    *  @return array<NDArray>
    *       outputs
    */
    public function backward(array $dOutputs) : array
    {
        $dInputs = $this->differentiate($dOutputs);
        return $dInputs;
    }
}
