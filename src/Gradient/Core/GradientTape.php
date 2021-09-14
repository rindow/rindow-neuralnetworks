<?php
namespace Rindow\NeuralNetworks\Gradient\Core;

use InvalidArgumentException;
use LogicException;
use Throwable;
use Rindow\NeuralNetworks\Support\Control\Context;
use Rindow\NeuralNetworks\Layer\LayerBase;
use Rindow\NeuralNetworks\Gradient\Core\Undetermined;

class GradientTape implements Context
{
    static public $autoBackProp = false;
    static public $debugBackward = null;
    static public $debug = false;

    protected $backend;
    protected $persistent;
    protected $backup;
    protected $grads = [];

    public function __construct($backend,$persistent=null)
    {
        $this->backend = $backend;
        $this->persistent = $persistent;
    }

    public function enter() : void
    {
        $this->backup = self::$autoBackProp;
        self::$autoBackProp = true;
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

        $targetId = spl_object_hash($target);
        if($this->persistent && array_key_exists($targetId,$this->grads)) {
            $grads = $this->grads[$targetId];
        } else {
            $grads = [];
            foreach($target->creator()->outputs() as $o) {
                $grads[$o->oid()] = $K->ones($o->shape(),$o->dtype());
            }
            //$grads[$targetId] = $K->onesLike($target->value());
        }

        if(!$this->persistent || !array_key_exists($targetId,$this->grads)) {
            $this->calcGradient($grads,$target);
        }
        foreach ($sources as $key => $source) {
            $sourceId = spl_object_hash($source);
            if(!array_key_exists($sourceId,$grads)) {
                throw new InvalidArgumentException("Invalid source variable");
            }
            $gradients[] = $grads[$sourceId];
        }
        if($this->persistent) {
            $this->grads[$targetId] = $grads;
        }

        if($singleValue) {
            return $gradients[0];
        }
        return $gradients;
    }

    protected function calcGradient(&$grads,$target) : void
    {
        $K = $this->backend;
        $funcs = [spl_object_hash($target->creator())=>$target->creator()];
        while(count($funcs)) {
            $func = array_pop($funcs);
            $dOutputs = [];
            foreach($func->outputs() as $o) {
                $oid = $o->oid();
                if(array_key_exists($oid,$grads)) {
                    $dOutputs[] = $grads[$oid];
                    // *** CAUTION ***
                    // Outputs are released as soon as the func object is
                    // released after being used in backwards.
                    // Index inconsistencies in grads occur because OIDs
                    // can be reused. Grads must be released to prevent
                    // this problem.
                    unset($grads[$oid]);
                } else {
                    $dOutputs[] = $K->zeros($o->shape(),$o->dtype());
                }
            }
            // with Config as tape:
            $tmpdInputs = $func->backward($dOutputs);
            unset($dOutputs);

            $dDatas = array_map(null,$func->inputs(),$tmpdInputs);
            unset($tmpdInputs);

            if($func instanceof LayerBase) {
                $dDatas = array_merge($dDatas,array_map(null,$func->weights(),$func->getGrads()));
            }

            foreach($dDatas as $dData) {
                [$inputs,$dInputs] = $dData;
                $sourceId = spl_object_hash($inputs);
                if(!array_key_exists($sourceId,$grads)) {
                    $grads[$sourceId] = $dInputs;
                } else {
                    $grads[$sourceId] = $K->add($grads[$sourceId],$dInputs);
                }

                $creator = $inputs->creator();
                if($creator) {
                    if(!array_key_exists(spl_object_hash($creator),$funcs)) {
                        $funcs[spl_object_hash($creator)] = $creator;
                        uasort($funcs,function($a,$b){return $a->generation()-$b->generation();});
                    }
                }
            }
        }
    }
}
