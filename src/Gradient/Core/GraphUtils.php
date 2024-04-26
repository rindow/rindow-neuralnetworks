<?php
declare(strict_types=1);
namespace Rindow\NeuralNetworks\Gradient\Core;

use InvalidArgumentException;
use Rindow\NeuralNetworks\Gradient\Variable as VariableInterface;
use WeakMap;
use ArrayAccess;

trait GraphUtils
{
    /**
     * @param array<VariableInterface> $graphOutputs
     * @return array{array<object>,array<object>,array<VariableInterface>}
     */
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
                if(!is_a($input,Variable::class)) {
                    $typename = is_object($input) ? get_class($input) : gettype($input);
                    throw new InvalidArgumentException("Invalid Argument for constant on ".$func->name().". gives $typename.");
                }
                $creator = $input->creator();
                if($creator!=null) {
                    //$oid = spl_object_id($creator);
                    if(!isset($used[$creator])) {
                        $used[$creator] = true;
                        $funcs[] = $creator;
                        usort($funcs,function($a,$b){return $a->generation()-$b->generation();});
                    }
                } else {
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
            $available = false;
            foreach($func->outputs() as $o) {
                $v = $o->get();
                if($v!=null && $v->isbackpropagatable() &&
                    isset($args[spl_object_id($v)])) {
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

    /**
     * @param array<object> $pipeline
     * @param ArrayAccess<object,object> $grads
     * @param array<object> $oidsToCollect
     */
    public function backwardPipeline(
        object $backend,
        array $pipeline, ArrayAccess $grads=null, array $oidsToCollect=null) : void
    {
        $K = $backend;
        foreach($pipeline as $func) {
            //echo "count(grads)=".count($grads)."\n";
            //echo "func=".basename(get_class($func))."\n";
            $dOutputs = [];
            foreach($func->outputs() as $o) {
                $oid = $o->get();
                if($oid!==null && isset($grads[$oid])) {
                    //echo 'grads('.spl_object_id($oid).')'."\n";
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
                    //echo 'grads('.spl_object_id($oid).') not found'."\n";
                    //$shape = $o->valueShape();
                    //$dtype = $o->dtype();
                    //array_unshift($shape,$batchSize);
                    //$dOutputs[] = $K->zeros($shape(),$dtype());
                    $dOutputs[] = $K->zeros($o->shape(),$o->dtype());
                }
            }
    
            $tmpdInputs = $func->backward($dOutputs,$grads,$oidsToCollect);
            //echo "after backword: count(grads)=".count($grads)."\n";
    
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
                    //echo "add grads(".spl_object_id($oid).")<=";
                    //echo "[".implode(',',$grads[$oid]->toArray())."]+";
                    //echo "[".implode(',',$dx->toArray())."]\n";
                    $grads[$oid] = $K->add($grads[$oid],$dx);
                } else {
                    //echo "set grads(".spl_object_id($oid).")\n";
                    $grads[$oid] = $dx;
                }
            }
        }
    }

    /**
     * @param array<mixed> $values
     * @return array<VariableInterface|null>
     */
    protected function packVariables(object $backend,array $values) : array
    {
        return array_map(function($value) use ($backend) {
            return ($value!==null)?new Variable($backend,$value):null;
        },$values);
    }

    /**
     * @param array<VariableInterface|null> $variables
     * @return array<mixed>
     */
    protected function unpackVariables(array $variables) : array
    {
        return array_map(function($variable){
            return ($variable instanceof Variable)?$variable->value():null;
        },$variables);
    }

    /**
     * @param array<VariableInterface|null> $variables
     * @return array<object|null>
     */
    protected function referenceVariables(array $variables) : array
    {
        return array_map(function($variable) {
            return ($variable!==null)?$variable->reference():null;
        },$variables);
    }

    /**
     * @param array<VariableInterface|null> $variables
     */
    protected function maxGeneration(array $variables) : int
    {
        return array_reduce($variables,function($max,$variable) {
            return ($variable!==null)?max($max,$variable->generation()):$max;},-1);
    }

    /**
     * @param array<VariableInterface|null> $variables
     * @return array<int|null>
     */
    protected function getObjectIds(array $variables) : array
    {
        return array_map(function($variable) {
            return ($variable!==null)?spl_object_id($variable):null;
        },$variables);
    }

    /**
     * @param array<VariableInterface> $variables
     * @return array<VariableInterface>
     */
    protected function repackVariables(object $backend,array $variables) : array
    {
        return array_map(function($variable) use ($backend) {
            return new Variable($backend,$variable);
        },$variables);
    }

    /**
     * @param array<VariableInterface> $variables
     */
    protected function setCreatorToVariables(object $creator,array $variables) : void
    {
        foreach($variables as $variable) {
            if($variable!==null) {
                $variable->setCreator($creator);
            }
        }
    }

}