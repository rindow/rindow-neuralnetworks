<?php
namespace Rindow\NeuralNetworks\Gradient\Core;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

abstract class AbstractFunction
{
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

    public function __construct($backend, array $options=null)
    {
        $this->backend = $backend;
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

        if(GradientTape::$autoBackProp) {
            $this->inputsVariables = $inputs;
        }
        if($inputs[0] instanceof Undetermined) {
            for($i=0;$i<$this->numOfOutputs;$i++) {
                $outputs[] = new Undetermined();
            }
        } else {
            $values = array_map(function($input){return $input->value();},$inputs);
            $outValues = $this->call($values);
            $outputs = array_map(function($value){return new Variable($this->backend,$value);},$outValues);
        }

        if(GradientTape::$autoBackProp) {
            $this->generation = array_reduce(
                $inputs,function($max,$x){return max($max,$x->generation());},-1);
            foreach ($outputs as $key => $output) {
                $output->setCreator($this);
            }
            $this->outputsVariables = array_map(function($o){return $o->reference();},$outputs);
        }
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
