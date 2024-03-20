<?php
declare(strict_types=1);
namespace Rindow\NeuralNetworks\Gradient\Core;

use InvalidArgumentException;
use stdClass;
use ArrayAccess;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Module;

class GraphSession
{
    static public $session;

    protected $backupSession = [];

    /**
    *  @var array<Variable>   inputs
    */
    protected $inputsVariables;

    /**
    *  @var Dict<Variable>   options
    */
    protected $optionsVariables = [];

    /**
    *  @var array<Variable>   outputs
    */
    protected $outputsVariables;

    /**
    *  @var int   generation
    */
    protected $generation;

    /**
    *  @var object $func   Shared Function Object. GraphFunction/Layer/Model etc.
    */
    protected $func;

    /**
    *  @var dict<stdClass>   container
    */
    protected $container = [];

    public function __construct(object $func,array $inputs, ?array $options=null)
    {
        $this->func = $func;
        $this->inputsVariables = $inputs;
        $this->generation = $this->maxGeneration($inputs);
        if($options!=null) {
            $this->optionsVariables = $options;
        }
    }

    public function name()
    {
        if(method_exists($this->func,'name')) {
            return $this->func->name();
        }
        return get_class($this->func);
    }

    protected function maxGeneration(array $variables)
    {
        return array_reduce($variables,function($max,$variable) {
            return ($variable!==null)?max($max,$variable->generation()):$max;},-1);
    }

    /**
    *  @return array<Variable>
    *       inputs
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
        return $this->optionsVariables;
    }

    public function _setOutputsVariables(array $outputs)
    {
        $this->outputsVariables = $outputs;
    }

    /**
    *  @return array<Variable>
    *       outputs
    */
    public function outputs() : array
    {
        return $this->outputsVariables;
    }

    public function _setGeneration($generation)
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

    public function begin()
    {
        array_push($this->backupSession,self::$session);
        self::$session = $this;
    }

    public function end()
    {
        self::$session = array_pop($this->backupSession);
    }

    public function _rawCall(array $inputs,array $options)
    {
        $this->begin();
        try {
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

    public function weights()
    {
        return $this->func->weights();
    }

    public function getGrads()
    {
        return $this->func->getGrads();
    }

    /**
    *  @return array<NDArray> dInputs
    *       function
    */
    public function backward(array $dOutputs, ArrayAccess $grads=null, array $oidsToCollect=null) : array
    {
        $this->begin();
        try {
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
