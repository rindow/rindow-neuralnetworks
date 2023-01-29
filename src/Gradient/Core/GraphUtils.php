<?php
declare(strict_types=1);
namespace Rindow\NeuralNetworks\Gradient\Core;

use InvalidArgumentException;
use Rindow\NeuralNetworks\Gradient\Module;
use WeakMap;
use ArrayAccess;

trait GraphUtils
{
    protected function buildPipeline(array $graphOutputs) : array
    {
        // compile forward
        $funcs = array_map(function($o){return $o->creator();},$graphOutputs);
        usort($funcs,function($a,$b){return $a->generation()-$b->generation();});
        $pipeline = [];
        $constants = [];
        $backprop = [];
        $used = new WeakMap();
        foreach($funcs as $func) {
            $used[$func] = true;
        }
        while(count($funcs)>0) {
            $func = array_pop($funcs);
            $pipeline[] = $func;
            $args = array_merge($func->inputs(),array_values($func->options()));
            foreach($args as $input) {
                $creator = $input->creator();
                if($creator!=null) {
                    //$oid = spl_object_id($creator);
                    if(!isset($used[$creator])) {
                        $used[$creator] = true;
                        $funcs[] = $creator;
                        usort($funcs,function($a,$b){return $a->generation()-$b->generation();});
                    }
                } else {
                    if($input===null) {
                        throw new InvalidArgumentException("Invalid Argument for constant on ".$func->name().". gived NULL");
                    }
                    $constants[] = $input;
                }
            }
        }

        // compile backward
        $args = [];
        foreach($graphOutputs as $o) {
            if($o instanceof VariableReference) {
                $o = $o->get();
            }
            $args[spl_object_id($o)] = true;
        }
        foreach($pipeline as $func) {
            if($func instanceof StopGradient) {
                continue;
            }
            $available = false;
            foreach($func->outputs() as $o) {
                if($o->get()!=null && isset($args[spl_object_id($o->get())])) {
                    $available = true;
                }
            }
            if($available) {
                foreach($func->inputs() as $o) {
                    $args[spl_object_id($o)] = true;
                }
                $backprop[] = $func;
            }
        }
        $pipeline = array_reverse($pipeline);
        return [$pipeline,$backprop,$constants];
    }

    public function backwardPipeline(
        object $backend,
        array $pipeline, ArrayAccess $grads=null, array $oidsToCollect=null) : void
    {
        $K = $backend;
        foreach($pipeline as $func) {
            $dOutputs = [];
            foreach($func->outputs() as $o) {
                $oid = $o->get();
                if($oid!==null && isset($grads[$oid])) {
                    $dOutputs[] = $grads[$oid];
                    // *** CAUTION ***
                    // Outputs are released as soon as the func object is
                    // released after being used in backwards.
                    // Index inconsistencies in grads occur because OIDs
                    // can be reused. Grads must be released to prevent
                    // this problem.
                    if(!is_array($oidsToCollect)) {
                        unset($grads[$oid]);
                    }
                } else {
                    //$shape = $o->valueShape();
                    //$dtype = $o->dtype();
                    //array_unshift($shape,$batchSize);
                    //$dOutputs[] = $K->zeros($shape(),$dtype());
                    $dOutputs[] = $K->zeros($o->shape(),$o->dtype());
                }
            }
    
            $tmpdInputs = $func->backward($dOutputs,$grads,$oidsToCollect);
    
            unset($dOutputs);

            $dDatas = array_map(null,$func->inputs(),$tmpdInputs);
            unset($tmpdInputs);

            foreach ($dDatas as $idx => [$input,$dx]) {
                if($dx===null) { // *** none Backpropagation ***
                    continue;
                }
                $oid = $input;
                if(isset($grads[$oid])) {
                    // **** CAUTION ****
                    // Must create new Instance of NDArray
                    // Don't use "update_add"!
                    // Because sometime grad and dx are same instace.
                    // Using update_add causes problems when branching function output more than once.
                    $grads[$oid] = $K->add($grads[$oid],$dx);
                } else {
                    $grads[$oid] = $dx;
                }
            }
        }
    }

    protected function packVariables(object $backend,array $values) : array
    {
        return array_map(function($value) use ($backend) {
            return ($value!==null)?new Variable($backend,$value):null;
        },$values);
    }

    protected function unpackVariables(array $variables) : array
    {
        return array_map(function($variable){
            return ($variable instanceof Variable)?$variable->value():null;
        },$variables);
    }

    protected function referenceVariables(array $variables) : array
    {
        return array_map(function($variable) {
            return ($variable!==null)?$variable->reference():null;
        },$variables);
    }

    protected function maxGeneration(array $variables)
    {
        return array_reduce($variables,function($max,$variable) {
            return ($variable!==null)?max($max,$variable->generation()):$max;},-1);
    }

    protected function getObjectIds(array $variables)
    {
        return array_map(function($variable) {
            return ($variable!==null)?spl_object_id($variable):null;
        },$variables);
    }

    protected function repackVariables(object $backend,array $variables) : array
    {
        return array_map(function($variable) use ($backend) {
            return new Variable($backend,$variable);
        },$variables);
    }

    protected function setCreatorToVariables(object $creator,array $variables) : void
    {
        foreach($variables as $variable) {
            if($variable!==null) {
                $variable->setCreator($creator);
            }
        }
    }

}