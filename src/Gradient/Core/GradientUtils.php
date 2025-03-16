<?php
declare(strict_types=1);
namespace Rindow\NeuralNetworks\Gradient\Core;

use InvalidArgumentException;
use stdClass;
use ArrayAccess;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Variable as VariableInterface;
use Rindow\NeuralNetworks\Gradient\Core\MaskedNDArray;
use Rindow\NeuralNetworks\Gradient\MaskedNDArray as MaskedNDArrayInterface;

trait GradientUtils
{
    protected ?stdClass $container;
    /** @var array<VariableInterface> $inputsVariables */
    protected array $inputsVariables;
    /** @var array<null|VariableReference> $outputsVariables */
    protected array $outputsVariables;
    protected int $generation;

    /**
     * @param array<VariableInterface> $inputsVariables
     * @param array<null|NDArray|VariableInterface> $outputs
     * @param array<bool> $unbackpropagatables
     * @return array<VariableInterface>
     */
    protected function postGradientProcess(
        object $backend, array $inputsVariables, array $outputs, ?array $unbackpropagatables=null) : array
    {
        $outputsVariables = [];
        foreach ($outputs as $key => $v) {
            $undetermined = ($v === null) ? true : false;
            $unbackpropagatable = isset($unbackpropagatables[$key]) && $unbackpropagatables[$key];
            $outputsVariables[] = new Variable(
                $backend, $v, undetermined:$undetermined, unbackpropagatable:$unbackpropagatable);
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

    /**
     * @param array<string,mixed> $optionsVariables
     * @return array<string,mixed>
     */
    protected function cleanNullValue(?array $optionsVariables) : array
    {
        if($optionsVariables!==null) {
            $keys = array_keys($optionsVariables);
            foreach ($keys as $key) {
                if($optionsVariables[$key]===null) {
                    unset($optionsVariables[$key]);
                }
            }
        }
        return $optionsVariables;
    }

    /**
     * @param array<VariableInterface> $inputsVariables
     * @param array<string,mixed> $optionsVariables
     */
    protected function preGradientProcessOnSession(
        array $inputsVariables, ?array $optionsVariables=null) : object
    {
        $session = new GraphSession($this,$inputsVariables,$optionsVariables);
        $session->_setGeneration($this->maxGeneration($inputsVariables));
        return $session;
    }

    /**
     * @param array<VariableInterface> $inputsVariables
     * @param array<null|NDArray|VariableInterface> $outputs
     * @param array<bool> $unbackpropagatables
     * @return array<VariableInterface>
     */
    protected function postGradientProcessOnSession(
        object $backend, object $session, array $inputsVariables,
        array $outputs, ?array $unbackpropagatables=null) : array
    {
        $outputsVariables = [];
        foreach ($outputs as $key => $v) {
            $undetermined = ($v === null) ? true : false;
            $unbackpropagatable = isset($unbackpropagatables[$key]) && $unbackpropagatables[$key];
            $outputsVariables[] = new Variable(
                $backend, $v, undetermined:$undetermined, unbackpropagatable:$unbackpropagatable);
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

    /**
     * @param array<array{object,NDArray}> $mapping
     * @param ArrayAccess<object,mixed> $grads
     * @param array<object> $oidsToCollect
     */
    protected function collectGradients(
        object $backend, array $mapping, ?ArrayAccess $grads=null, ?array $oidsToCollect=null) : void
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

    protected function packVariable(object $backend, mixed $value) : mixed
    {
        if($value instanceof VariableInterface) {
            return $value;
        } 
        return new Variable($backend,$value);
    }

    /**
     * @param array<mixed> $values
     * @return array<null|VariableInterface>
     */
    protected function packVariables(object $backend,array $values) : array
    {
        return array_map(function($value) use ($backend) {
            return ($value!==null)?new Variable($backend,$value):null;
        },$values);
    }

    protected function unpackVariable(object $backend, mixed $value) : mixed
    {
        if($value instanceof Variable) {
            return $value->value();
        } 
        return $value;
    }

    /**
     * @return array{null|VariableInterface,mixed}
     */
    public function packAndUnpackVariable(
        object $backend, mixed $value, ?bool $unbackpropagatable=null) : array
    {
        if($value===null) {
            return [null,null];
        }
        if($value instanceof Variable) {
            $rawValue = $value->value();
        } else {
            $rawValue = $value;
            $value = new Variable($backend,$rawValue,unbackpropagatable:$unbackpropagatable);
        }
        return [$value,$rawValue];
    }

    /**
     * @param array<mixed> $values
     * @return array{array<null|VariableInterface>,array<mixed>}
     */
    public function packAndUnpackVariables(
        object $backend, array $values, ?bool $unbackpropagatable=null) : array
    {
        $variables = [];
        $rawValues = [];
        foreach($values as $value) {
            if($value===null) {
                $variables[] = null;
                $rawValues[] = null;
            } elseif($value instanceof Variable) {
                $variables[] = $value;
                $rawValues[] = $value->value();
            } else {
                $variables[] = new Variable($backend,$value,unbackpropagatable:$unbackpropagatable);
                $rawValues[] = $value;
            }
        }
        return [$variables,$rawValues];
    }

    /**
     * @param array<null|VariableInterface> $variables
     * @return array<null|VariableReference>
     */
    protected function referenceVariables(array $variables) : array
    {
        return array_map(function($variable) {
            return ($variable!==null)?$variable->reference():null;
        },$variables);
    }

    /**
     * @param array<null|VariableInterface> $variables
     */
    protected function maxGeneration(array $variables) : int
    {
        return array_reduce($variables,function($max,$variable) {
            return ($variable!==null)?max($max,$variable->generation()):$max;},-1);
    }

    /**
     * @param array<null|VariableInterface> $variables
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
