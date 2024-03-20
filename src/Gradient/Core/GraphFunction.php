<?php
declare(strict_types=1);
namespace Rindow\NeuralNetworks\Gradient\Core;

use InvalidArgumentException;
use RuntimeException;
use Throwable;
use ArrayAccess;
use WeakMap;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\Control\Execute;
use Rindow\NeuralNetworks\Gradient\Module;
use Rindow\NeuralNetworks\Gradient\Scalar as ScalarInterface;
use Rindow\NeuralNetworks\Gradient\ArrayShape as ArrayShapeInterface;

class GraphFunction
{
    use GraphUtils;

    const EAGER_EXECUTION = 0;
    const UNDER_CONSTRUCTION = 1;
    const EXECUTING = 2;

    static public $mode = self::EAGER_EXECUTION;

    protected $backupMode = [];

    /**
    *  @var object   backend
    */
    protected $backend;

    /**
    *  @var callable func
    */
    protected $func;

    /**
    *  @var bool built
    */
    protected $built = false;

    /**
    *  @var int   numOfInputs
    */
    protected $numOfInputs;

    /**
    *  @var int   numOfOutputs
    */
    protected $numOfOutputs;

    protected $startInputOids;

    protected $endOutputOids;

    /**
    *  @var array<AbstractFunction>   graph forward pipeline
    */
    protected $pipeline;

    /**
    *  @var array<AbstractFunction>   graph backward pipeline
    */
    protected $backprop;

    /**
    *  @var Dict<NDArray>    constants for input oid in the graph
    */
    protected $constants;

    protected $alternateCreator;

    public function __construct(object $backend, callable $func, object $alternateCreator=null)
    {
        $this->backend = $backend;
        $this->func = $func;
        $this->alternateCreator = $alternateCreator;
    }

    public function backend()
    {
        return $this->backend;
    }

    protected function executeOnMode($sessionFunc,int $mode,callable $func)
    {
        array_push($this->backupMode,self::$mode);
        self::$mode = $mode;
        try {
            $sessionFunc->begin();
            try {
                $outputs = $func();
            } catch(Throwable $e) {
                $sessionFunc->end();
                throw $e;
            }
            $sessionFunc->end();
        } catch(Throwable $e) {
            self::$mode = array_pop($this->backupMode);
            throw $e;
        }
        self::$mode = array_pop($this->backupMode);
        return $outputs;
    }

    /**
    *  @param array<Variable>  $inputs
    *       inputs
    *  @return array<Variable>
    *       outputs
    */
    public function __invoke(...$inputs)
    {
        if(!$this->built) {
            return $this->build($inputs);
        }
        $creator = $this->alternateCreator ?? $this;
        $sessionFunc = new GraphSession($creator,$inputs);
        if(count($inputs)!=$this->numOfInputs) {
            throw new InvalidArgumentException($this->numOfInputs.' arguments are required.');
        }
        if(self::$mode!=self::EXECUTING) {
            if(GradientTape::$autoBackProp) {
                $sessionFunc->_setGeneration($this->maxGeneration($inputs));
            }
            $inputs = $this->unpackVariables($inputs);
        }

        // execute graph 
        $outValues = $this->executeOnMode($sessionFunc,self::EXECUTING,function() use ($inputs) {
            return $this->_rawCall($inputs,[]);
        });

        // finalize outputs
        if(self::$mode!=self::EXECUTING) {
            $outputs = $this->packVariables($this->backend,$outValues);
            if(GradientTape::$autoBackProp) {
                $this->setCreatorToVariables($sessionFunc,$outputs);
                $sessionFunc->_setOutputsVariables($this->referenceVariables($outputs));
            }
        }
        if(count($outputs)==1) {
            $outputs = $outputs[0];
        }
        return $outputs;
    }

    public function _rawCall(array $inputs,array $options) : array
    {
        $K = $this->backend;
        $vars = new WeakMap();
        foreach($this->constants as $inp) {
            $vars[$inp] = $inp->value();
        }
        foreach(array_map(null,$this->startInputOids,$inputs) as [$oid,$inp]) {
            $vars[$oid] = $inp;
        }
        
        $funcs = $this->pipeline;

        foreach($funcs as $func) {
            $oids = $func->inputs();
            $inps = array_map(function($oid) use ($vars) {return $vars[$oid];}, $oids);
            $opts = [];
            foreach ($func->options() as $key => $variable) {
                $opts[$key] = $vars[$variable] ?? null;
            }
            $outs = $func->_rawCall($inps,$opts);
            foreach(array_map(null,$func->outputs(),$outs) as [$o,$out]) {
                $oid = $o->get();
                if($oid!==null) {
                    $vars[$oid] = $out;
                }
                if($out instanceof ScalarInterface) {
                    $o->_setShape([]);
                } elseif($out instanceof ArrayShapeInterface) {
                    $o->_setShape([count($out)]);
                } elseif($out instanceof NDArray) {
                    $o->_setShape($out->shape());
                } else {
                    echo get_class($func)."\n";
                    var_dump($func->outputs());
                    echo "--------\n";
                    var_dump($outs);
                    throw new LogicException('Invalid data type.');
                }
            }
        }

        $outValues = [];
        foreach ($this->endOutputOids as $oid) {
            $outValues[] = $vars[$oid];
        }
        return $outValues;
    }

    protected function build(array $inputs)
    {
        $K = $this->backend;
        $creator = $this->alternateCreator ?? $this;
        $sessionFunc = new GraphSession($creator,$inputs);
        $sessionFunc->_setGeneration($this->maxGeneration($inputs));
        $inputs = $this->repackVariables($this->backend,$inputs);
        $this->startInputOids = $inputs;

        // build graph
        $this->numOfInputs = count($inputs);
        $func = $this->func;
        $graphOutputs = Execute::with(new GradientTape($this->backend),function() use ($sessionFunc,$func,$inputs) {
            return $this->executeOnMode($sessionFunc,self::UNDER_CONSTRUCTION,function() use ($func,$inputs) {
                return $func(...$inputs);
            });
        });
        if(!is_array($graphOutputs)) {
            $graphOutputs = [$graphOutputs];
        }

        $this->endOutputOids = $graphOutputs;

        [$pipeline,$backprop,$tmpconsts] = $this->buildPipeline($graphOutputs);
        $constants = [];
        foreach($tmpconsts as $c) {
            if(!in_array($c,$this->startInputOids,true)) {
                $constants[] = $c;
            }
        }
        unset($tmpconsts);
        $this->constants = $constants; // NDArray
        $this->pipeline = $pipeline; // Func
        $this->backprop = $backprop; // Func
        $this->built = true;
        $outputs = $this->repackVariables($this->backend,$graphOutputs);
        foreach($pipeline as $func) {
            foreach($func->inputs() as $o) {
                if($o->creator()!==null || in_array($o,$this->startInputOids,true)) {
                    // Clearing variables without constants and weights.
                    // Because It wants to save the math buffer.
                    $o->_clearValue();
                }
            }
        }
        $this->setCreatorToVariables($sessionFunc,$outputs);
        $sessionFunc->_setOutputsVariables($this->referenceVariables($outputs));
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
    public function backward(array $dOutputs, ArrayAccess $grads=null, array $oidsToCollect=null) : array
    {
        if(!$this->built) {
            throw new RuntimeException('Not yet built');
        }
        $K = $this->backend;
        //$backupGradOids = array_keys($grads);
        foreach(array_map(null,$this->endOutputOids,$dOutputs) as $oset) {
            [$oid,$dOut] = $oset;
            $grads[$oid] = $dOut;
            //echo "set grads(".spl_object_id($oid).") from endOutputs\n";
        }
        unset($output);
        unset($dOut);
        unset($oset);
        unset($dOutputs);

        $backprop = $this->backprop;
        $this->backwardPipeline($this->backend, $backprop, $grads, $oidsToCollect);

        foreach($this->startInputOids as $oid) {
            if(!isset($grads[$oid])) {
                //throw new InvalidArgumentException("Invalid input variables");
                $dInputs[] = null; // maybe, it is skiped argument in internal.
                continue;
            }
            $dInputs[] = $grads[$oid];
        }
        // Like WeakMap
        //$unsets = [];
        //foreach ($grads as $oid => $value) {
        //    if(!in_array($oid,$oidsToCollect,true) && !in_array($oid,$backupGradOids,true)) {
        //        $unsets[] = $oid;
        //    }
        //}
        //foreach ($unsets as $oid) {
        //    unset($grads[$oid]);
        //}
        return $dInputs;
    }

}
