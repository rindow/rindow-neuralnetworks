<?php
namespace Rindow\NeuralNetworks\Gradient\Core;

use InvalidArgumentException;

trait GradientUtils
{
    protected function postGradientProcess(
        $backend, array $inputsVariables, array $outputs) : array
    {
        $outputsVariables = [];
        foreach ($outputs as $v) {
            if($v === null) {
                $outputsVariables[] = new Undetermined();
            } else {
                $outputsVariables[] = new Variable($backend,$v);
            }
        }
        if(GradientTape::$autoBackProp) {
            $this->inputsVariables = $inputsVariables;
            $this->generation = array_reduce(
                $inputsVariables,function($max,$x){return max($max,$x->generation());},-1);
            foreach ($outputsVariables as $o) {
                $o->setCreator($this);
            }
            $this->outputsVariables = array_map(
                function($o){return $o->reference();},$outputsVariables);
        }
        return $outputsVariables;
    }
}
