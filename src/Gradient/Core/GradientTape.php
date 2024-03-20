<?php
declare(strict_types=1);
namespace Rindow\NeuralNetworks\Gradient\Core;

use InvalidArgumentException;
use LogicException;
use Throwable;
use WeakMap;
use Rindow\NeuralNetworks\Support\Control\Context;
use Rindow\NeuralNetworks\Layer\LayerBase;
use Rindow\NeuralNetworks\Gradient\Variable as VariableInterface;

class GradientTape implements Context
{
    use GraphUtils;

    static public $autoBackProp = null;
    static public $debugBackward = null;
    static public $debug = false;

    protected $backend;
    protected $persistent;
    protected $backup;
    protected $persistentGrads = [];
    protected $lockingObjects = [];

    public function __construct(object $backend,bool $persistent=null)
    {
        $this->backend = $backend;
        $this->persistent = $persistent;
    }

    public function enter() : void
    {
        $this->backup = self::$autoBackProp;
        self::$autoBackProp = $this;
    }

    public function exit(Throwable $e=null) : bool
    {
        self::$autoBackProp = $this->backup;
        return false;
    }

    public function gradient(VariableInterface $target,$sources)
    {
        if(self::$autoBackProp) {
            throw new LogicException("The gradient function is not supported for use within the automatic differentiation context.");
        }
        $K = $this->backend;
        $singleValue = false;
        if($target->creator()==null)
            return null;
        if(!is_array($sources)) {
            $singleValue = true;
            $sources = [$sources];
        }
        $gradients = [];

        $targetId = spl_object_id($target);
        if($this->persistent && array_key_exists($targetId,$this->persistentGrads)) {
            $grads = $this->persistentGrads[$targetId];
        } else {
            $grads = new WeakMap();
            foreach($target->creator()->outputs() as $o) {
                if($o->get()===$target) {
                    //echo "set grads(".spl_object_id($o->get()).") <= Ones from target func's outputs\n";
                    $grads[$o->get()] = $K->ones($o->shape(),$o->dtype());
                } else {
                    //echo "set grads(".spl_object_id($o->get()).") <= Zeros from target func's outputs\n";
                    $grads[$o->get()] = $K->zeros($o->shape(),$o->dtype());
                }
            }
            //$grads[$targetId] = $K->onesLike($target->value());
        }

        $sourceIds = $sources;
        if(!$this->persistent || !array_key_exists($targetId,$this->persistentGrads)) {
            $this->calcGradient($grads,$target,$sourceIds);
        }
        $idx = 1;
        foreach ($sourceIds as $sourceId) {
            if(!isset($grads[$sourceId])) {
                $name = $sourceId->name();
                throw new InvalidArgumentException("No applicable gradient found for source #{$idx}({$name}).");
            }
            $gradients[] = $grads[$sourceId];
            $idx++;
        }
        if($this->persistent) {
            $this->persistentGrads[$targetId] = $grads;
        }

        if($singleValue) {
            return $gradients[0];
        }
        return $gradients;
    }

    protected function calcGradient($grads,$target,$sourceIds) : void
    {
        $graphOutputs = [$target];
        [$pipeline,$backprop,$constants] = $this->buildPipeline($graphOutputs);
        $this->backwardPipeline($this->backend,$backprop,$grads,$sourceIds);
    }
}
