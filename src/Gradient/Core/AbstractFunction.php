<?php
declare(strict_types=1);
namespace Rindow\NeuralNetworks\Gradient\Core;

use InvalidArgumentException;
use stdClass;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Scalar as ScalarInterface;
use Rindow\NeuralNetworks\Gradient\ArrayShape as ArrayShapeInterface;
use Rindow\NeuralNetworks\Gradient\Variable as VariableInterface;
use Rindow\NeuralNetworks\Gradient\Core\Scalar;

abstract class AbstractFunction
{
    use GradientUtils;
    /** @var array<bool> */
    protected ?array $unbackpropagatables = null;

    /**
    *  @param array<NDArray>  $inputs
    *  @return array<NDArray>
    */
    abstract protected function call(array $inputs) : array;

    /**
    *  @param array<NDArray>  $dOutputs
    *  @return array<NDArray|ScalarInterface>
    */
    abstract protected function differentiate(array $dOutputs) : array;

    protected object $backend;

    /** @var array<VariableInterface> $inputsVariables */
    protected array $inputsVariables;
    /** @var array<null|VariableReference> $outputsVariables */
    protected array $outputsVariables;
    protected ?string $name;

    protected int $generation;

    protected int $numOfInputs = 1;

    protected int $numOfOutputs = 1;

    protected ?stdClass $container;

    public function __construct(object $backend, ?string $name=null)
    {
        $this->backend = $backend;
        $this->name = $name;
        $this->container = new stdClass();
    }

    /**
    *  @return array<VariableInterface>
    */
    public function inputs() : array
    {
        return $this->inputsVariables;
    }

    /**
    *  @return array<VariableInterface>
    */
    public function options() : array
    {
        return [];
    }

    /**
    *  @return array<VariableReference>
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

    public function name() : ?string
    {
        return $this->name;
    }

    /**
     * Call from SessionFunc in compiled graph
     * @param array<NDArray> $inputs
     * @param array<string,mixed> $options
     * @return array<NDArray>
     */
    public function _rawCall(array $inputs,array $options) : array
    {
        return $this->call($inputs);
    }

    public function className() : string
    {
        return get_class($this);
    }

    /**
     * @param array<mixed> $inputs
     * @return array<mixed>
     */
    protected function preprocess(array $inputs) : array
    {
        return $inputs;
    }

    protected function toScalar(mixed $value,int $argIdx) : mixed
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

    /**
     * @return array<ArrayShapeInterface|VariableInterface>
     */
    protected function extractShapeArgment(mixed $shape) : array
    {
        if($shape instanceof VariableInterface) {
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

    /**
     * @param array<mixed> $inputs
     * @return array<int>
     */
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

    public function __invoke(mixed ...$inputs) : mixed
    {
        if(count($inputs)!=$this->numOfInputs) {
            throw new InvalidArgumentException($this->numOfInputs.' arguments are required.'.count($inputs).' given.'); 
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
    *  @param array<NDArray>  $dOutputs
    *  @return array<NDArray>
    */
    public function backward(array $dOutputs) : array
    {
        $dInputs = $this->differentiate($dOutputs);
        return $dInputs;
    }
}
