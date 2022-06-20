<?php
declare(strict_types=1);
namespace Rindow\NeuralNetworks\Gradient\Core;

use InvalidArgumentException;
use stdClass;
use Interop\Polite\Math\Matrix\NDArray;

abstract class AbstractFunction
{
    use GradientUtils;
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
        $values = array_map(function($input){return $input->value();},$inputs);
        $outValues = $this->call($values);
        $outputs = $this->postGradientProcess($this->backend,$inputs,$outValues);

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
