<?php
declare(strict_types=1);
namespace Rindow\NeuralNetworks\Gradient\Core;

use InvalidArgumentException;
use stdClass;
use ArrayAccess;

trait GradientUtils
{
    protected $container;

    protected function postGradientProcess(
        $backend, array $inputsVariables, array $outputs) : array
    {
        $outputsVariables = [];
        foreach ($outputs as $v) {
            $undetermined = ($v === null) ? true : false;
            $outputsVariables[] = new Variable(
                $backend, $v, undetermined:$undetermined);
        }
        if(GradientTape::$autoBackProp) {
            $this->inputsVariables = $inputsVariables;
            $this->generation = $this->maxGeneration($inputsVariables);
            foreach ($outputsVariables as $o) {
                $o->setCreator($this);
            }
            $this->outputsVariables = $this->referenceVariables($outputsVariables);
        }
        return $outputsVariables;
    }

    protected function preGradientProcessOnSession($inputsVariables,$optionsVariables=null) : object
    {
        $session = new GraphSession($this,$inputsVariables,$optionsVariables);
        $session->_setGeneration($this->maxGeneration($inputsVariables));
        return $session;
    }

    protected function postGradientProcessOnSession(
        object $backend, object $session, array $inputsVariables, array $outputs) : array
    {
        $outputsVariables = [];
        foreach ($outputs as $v) {
            $undetermined = ($v === null) ? true : false;
            $outputsVariables[] = new Variable(
                $backend, $v, undetermined:$undetermined);
        }
        if(GradientTape::$autoBackProp) {
            $this->setCreatorToVariables($session,$outputsVariables);
            $session->_setOutputsVariables($this->referenceVariables($outputsVariables));
        }
        return $outputsVariables;
    }

    protected function container() : object
    {
        $session = GraphSession::$session;
        if($session==null) {
            if($this->container===null) {
                $this->container = new stdClass();
            }
            return $this->container;
        }
        return $session->localContainer($this);
    }

    protected function collectGradients(object $backend, array $mapping, ArrayAccess $grads=null, array $oidsToCollect=null) : void
    {
        if($oidsToCollect===null) {
            return;
        } elseif(count($oidsToCollect)==0) {
            return;
        }
        $K = $backend;
        foreach($mapping as [$w,$g]) {
            $oid = $w;
            if(in_array($oid,$oidsToCollect,true)) {
                if(isset($grads[$oid])) {
                    $grads[$oid] = $K->add($grads[$oid],$g);
                } else {
                    $grads[$oid] = $g;
                }
            }
        }
    }

    protected function packVariable(object $backend, $value)
    {
        if($value instanceof Variable) {
            return $value;
        } 
        return new Variable($backend,$value);
    }

    protected function packVariables(object $backend,array $values) : array
    {
        return array_map(function($value) use ($backend) {
            return ($value!==null)?new Variable($backend,$value):null;
        },$values);
    }

    protected function unpackVariable(object $backend, $value)
    {
        if($value instanceof Variable) {
            return $value->value();
        } 
        return $value;
    }

    public function packAndUnpackVariable(object $backend, $value) : array
    {
        if($value instanceof Variable) {
            $rawValue = $value->value();
        } else {
            $rawValue = $value;
            $value = new Variable($backend,$rawValue);
        }
        return [$value,$rawValue];
    }

    public function packAndUnpackVariables(object $backend, array $values) : array
    {
        $variables = [];
        $rawValues = [];
        foreach($values as $value) {
            if($value instanceof Variable) {
                $variables[] = $value;
                $rawValues[] = $value->value();
            } else {
                $variables[] = new Variable($backend,$value);
                $rawValues[] = $value;
            }
        }
        return [$variables,$rawValues];
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

    protected function setCreatorToVariables(object $creator,array $variables) : void
    {
        foreach($variables as $variable) {
            if($variable!==null) {
                $variable->setCreator($creator);
            }
        }
    }
}
