<?php
declare(strict_types=1);
namespace Rindow\NeuralNetworks\Gradient\Core;

use InvalidArgumentException;
use LogicException;
use Throwable;
use WeakMap;
use Rindow\NeuralNetworks\Support\Control\Context;
use Rindow\NeuralNetworks\Layer\LayerBase;

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

    public function gradient($target,$sources)
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
                $grads[$o->get()] = $K->ones($o->shape(),$o->dtype());
            }
            //$grads[$targetId] = $K->onesLike($target->value());
        }

        $sourceIds = $sources;
        if(!$this->persistent || !array_key_exists($targetId,$this->persistentGrads)) {
            $this->calcGradient($grads,$target,$sourceIds);
        }
        foreach ($sourceIds as $sourceId) {
            if(!isset($grads[$sourceId])) {
                throw new InvalidArgumentException("No applicable gradient found for source");
            }
            $gradients[] = $grads[$sourceId];
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
