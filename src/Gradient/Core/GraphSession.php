<?php
declare(strict_types=1);
namespace Rindow\NeuralNetworks\Gradient\Core;

use InvalidArgumentException;
use stdClass;
use ArrayAccess;
use Throwable;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Layer\Layer;
use Rindow\NeuralNetworks\Model\Model;
use Rindow\NeuralNetworks\Loss\Loss;

class GraphSession
{
    static public ?object $session=null;
    /** @var array<object> $backupSession */
    protected array $backupSession = [];
    /** @var array<Variable> $inputsVariables */
    protected array $inputsVariables;
    /** @var array<string,Variable> $optionsVariables */
    protected array $optionsVariables = [];
    /** @var array<Variable> $outputsVariables */
    protected array $outputsVariables;
    protected int $generation;
    protected GraphFunction|Layer|Model|Loss $func;
    /** @var array<stdClass> $container */
    protected array $container = [];

    /**
    *  @param array<Variable> $inputs
    *  @param array<string,mixed> $options
    */
    public function __construct(object $func,array $inputs, ?array $options=null)
    {
        $this->func = $func;
        $this->inputsVariables = $inputs;
        $this->generation = $this->maxGeneration($inputs);
        if($options!=null) {
            $this->optionsVariables = $options;
        }
    }

    public function name() : ?string
    {
        // this func is model
        if(method_exists($this->func,'name')) {
            return $this->func->name();
        }
        return get_class($this->func);
    }

    /**
    *  @param array<null|Variable> $variables
    */
    protected function maxGeneration(array $variables) : int
    {
        return array_reduce($variables,function($max,$variable) {
            return ($variable!==null)?max($max,$variable->generation()):$max;},-1);
    }

    /**
    *  @return array<Variable>
    */
    public function inputs() : array
    {
        return $this->inputsVariables;
    }

    /**
    *  @return array<string,Variable>
    */
    public function options() : array
    {
        return $this->optionsVariables;
    }

    /**
    *  @param array<Variable> $outputs
    */
    public function _setOutputsVariables(array $outputs) : void
    {
        $this->outputsVariables = $outputs;
    }

    /**
    *  @return array<Variable>
    */
    public function outputs() : array
    {
        return $this->outputsVariables;
    }

    public function _setGeneration(int $generation) : void
    {
        $this->generation = $generation;
    }

    /**
    *  @return int
    */
    public function generation() : int
    {
        return $this->generation;
    }

    public function begin() : void
    {
        array_push($this->backupSession,self::$session);
        self::$session = $this;
    }

    public function end() : void
    {
        self::$session = array_pop($this->backupSession);
    }

    /**
     * @param array<NDArray> $inputs
     * @param array<string,mixed> $options
     * @return array<NDArray>
     */
    public function _rawCall(array $inputs,array $options) : array
    {
        $this->begin();
        try {
            // this func is a function or layer or loss or model
            $outputs = $this->func->_rawCall($inputs,$options);
        } catch(Throwable $e) {
            $this->end();
            throw $e;
        }
        $this->end();
        return $outputs;
    }

    public function className() : string
    {
        return get_class($this->func);
    }

    /**
    *  @return array<Variable>
    */
    public function weights() : array
    {
        // this func is a layer only
        return $this->func->weights();
    }

    /**
     * @return array<NDArray>
     */
    public function getGrads()
    {
        // this func is a layer only
        return $this->func->getGrads();
    }

    /**
     * @param array<NDArray> $dOutputs
     * @param ArrayAccess<object,object> $grads
     * @param array<object> $oidsToCollect
     * @return array<NDArray>
     */
    public function backward(
        array $dOutputs,
        ArrayAccess $grads=null,
        array $oidsToCollect=null
        ) : array
    {
        $this->begin();
        try {
            // this func is a activation or gradientfunc or layer or loss or model
            $dInputs = $this->func->backward($dOutputs, $grads, $oidsToCollect);
        } catch(Throwable $e) {
            $this->end();
            throw $e;
        }
        $this->end();
        if(count($this->backupSession)==0) {
            $this->container = [];
        }
        return $dInputs;
    }

    public function func() : object
    {
        return $this->func;
    }

    public function localContainer(object $object) : object
    {
        $oid = spl_object_id($object);
        if(!isset($this->container[$oid])) {
            $this->container[$oid] = new stdClass();
        }
        return $this->container[$oid];
    }
}
