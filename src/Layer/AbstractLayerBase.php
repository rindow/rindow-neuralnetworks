<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use LogicException;
use ArrayAccess;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Activation\FunctionFactory;
use Rindow\NeuralNetworks\Activation\Activation as ActivationInterface;
use Rindow\NeuralNetworks\Gradient\Core\Variable;
use Rindow\NeuralNetworks\Layer\Embedding;
/**
 *
 */
abstract class AbstractLayerBase implements Layer
{
    protected $inputShape;
    protected $outputShape;
    protected $statesShapes;
    protected $activation;
    protected $activationName;
    protected $params=[];
    protected $grads=[];
    protected $shapeInspection = true;
    protected $name;
    // dynamic step interfaces
    //protected $inputsVariables;
    //protected $outputsVariables;
    //protected $generation;
    protected $weights=[];
    protected $assignedWeights = false;
    protected $built = false;
    protected $training = false;
    protected $callOptions = [];

    public function isBuilt() : bool
    {
        return $this->built;
    }

    public function getActivation()
    {
        return $this->activation;
    }

    public function setActivation(
        $activation) : void
    {
        if($activation==null){
            return;
        }
        if(is_string($activation)) {
            $this->activation = FunctionFactory::factory($this->backend,$activation);
            $this->activationName = $activation;
            return;
        }
        if($activation instanceof ActivationInterface) {
            $this->activation = $activation;
            // for compiling lossfunction
            if($this->activationName==null){
                $this->activationName = get_class($activation);
            }
            return;
        }
        throw new InvalidArgumentException('activation function must have the Activation interface');
    }

    protected function createFunction(
        string $activation=null)
    {
        if($activation==null){
            return null;
        }
        return FunctionFactory::factory($this->backend,$activation);
    }

    public function build($variable=null, array $sampleWeights=null)
    {
        $inputShape = $this->normalizeInputShape($variable);
        if($inputShape!==null)
            $this->inputShape = $inputShape;
        $this->outputShape = $inputShape;
    }

    protected function normalizeInputShape($variables=null) : array
    {
        if($variables===null) {
            $inputShape = $this->inputShape;
        } elseif($variables instanceof Variable) {
            $inputShape = $variables->valueShape();
            if($inputShape===null) {
                $inputShape = $this->inputShape;
            }
        } elseif(is_array($variables)) {
            $inputShape = [];
            foreach($variables as $idx => $v) {
                if(!($v instanceof Variable)) {
                    throw new InvalidArgumentException('variable list must contain Variables: "'.$this->typename($v).'" included in #'.$idx.'.');
                }
                $inputShape[] = $v->valueShape();
            }
        } else {
            throw new InvalidArgumentException('variable must be Variable type or null. "'.$this->typename($variables).'" given.');
        }

        if($inputShape===null) {
            throw new InvalidArgumentException("inputShape must be spacified");
        }
        return $this->fixInputShape($inputShape);
    }

    protected function normalizeCellInputShape(array $inputShape=null) : array
    {
        if($inputShape==null) {
            $inputShape = $this->inputShape;
        }
        return $this->fixInputShape($inputShape);
    }

    protected function fixInputShape($inputShape) : array
    {
        if($this->inputShape===null) {
            $this->inputShape = $inputShape;
        }
        if($this->shapeInspection && $this->inputShape!==$inputShape) {
            if(is_array($this->inputShape)) {
                $msg = 'Input shape is inconsistent: defined as '.$this->shapeToString($this->inputShape).
                ' but '.$this->shapeToString($inputShape).' given in '.$this->basename($this);
            } else {
                $msg = 'Input shape is inconsistent';
            }
            throw new InvalidArgumentException($msg);
        } elseif($this->inputShape===null && $inputShape===null) {
            throw new InvalidArgumentException('Input shape is not defined');
        }
        //$this->inputShape = $inputShape;
        return $inputShape;
    }

    public function inputShape() : array
    {
        return $this->inputShape;
    }

    public function outputShape() : array
    {
        return $this->outputShape;
    }

    public function statesShapes() : array
    {
        return $this->statesShapes;
    }

    public function getParams() : array
    {
        return $this->params;
    }

    public function getGrads() : array
    {
        return $this->grads;
    }

    public function reverseSyncWeightVariables() : void
    {
    }

    public function getConfig() : array
    {
        return [];
    }

    public function setName(string $name) : void
    {
        $this->name = $name;
    }

    public function getName() : string
    {
        return $this->name;
    }

    public function setShapeInspection(bool $enable)
    {
        $this->shapeInspection = $enable;
    }

    protected function shapeToString($shape)
    {
        if(!is_array($shape)) {
            return strval($shape);
        }
        $string = '[';
        foreach($shape as $value) {
            if($string!='[') {
                $string .= ',';
            }
            $string .= $this->shapeToString($value);
        }
        $string .= ']';
        return $string;
    }

    protected function assertInputShape(NDArray $inputs,$direction)
    {
        if(!$this->shapeInspection)
            return;
        if($this->inputShape===null) {
            throw new InvalidArgumentException('Uninitialized input shape');
        }
        $shape = $inputs->shape();
        $batchNum = array_shift($shape);
        if($shape!=$this->inputShape) {
            $shape = $this->shapeToString($shape);
            $inputShape = $this->shapeToString($this->inputShape);
            $name = $this->name ?? $this->basename($this);
            throw new InvalidArgumentException('unmatch input shape: '.$shape.', must be '.$inputShape.' in '.$name.':'.$direction);
        }
    }

    protected function assertOutputShape(NDArray $outputs,$direction)
    {
        if(!$this->shapeInspection)
            return;
        if($this->outputShape===null) {
            throw new InvalidArgumentException('Uninitialized output shape');
        }
        $shape = $outputs->shape();
        $batchNum = array_shift($shape);
        if($shape!=$this->outputShape) {
            $shape = $this->shapeToString($shape);
            $outputShape = $this->shapeToString($this->outputShape);
            $name = $this->name ?? $this->basename($this);
            throw new InvalidArgumentException('unmatch output shape: '.$shape.', must be '.$outputShape.' in '.$name.':'.$direction);
        }
    }

    protected function assertStatesShape(array $states=null,$direction)
    {
        if(!$this->shapeInspection)
            return;
        if($states===null)
            return;
        if($this->statesShapes===null) {
            throw new InvalidArgumentException('Uninitialized status shape');
        }
        if(count($states)!=count($this->statesShapes)){
            throw new InvalidArgumentException('Unmatch num of status. status need '.count($this->statesShapes).' NDArray. '.count($states).'given.');
        }
        foreach($states as $idx=>$state){;
            $stateShape = $this->statesShapes[$idx];
            $shape = $state->shape();
            $batchNum = array_shift($shape);
            if($shape!=$stateShape) {
                $shape = $this->shapeToString($shape);
                $stateShape = $this->shapeToString($stateShape);
                $name = $this->name ?? $this->basename($this);
                throw new InvalidArgumentException('Shape of state'.$idx.' must be '.$stateShape.', '.$shape.' given in '.$name.':'.$direction);
            }
        }
    }

    protected function allocateWeights($num,int $nonTrainables=null) : void
    {
        $variables = [];
        for($i=0;$i<$num;$i++) {
            $variables[] = new Variable($this->backend, null, undetermined: true);
        }
        if($nonTrainables) {
            for($i=0;$i<$nonTrainables;$i++) {
                $variables[] = new Variable(
                    $this->backend,null, undetermined: true, trainable: false);
            }
        }
        $this->weights = $variables;
    }

    /**
    *  @return array<Variable>
    */
    public function weights() : array
    {
        return $this->weights;
    }

    public function syncWeightVariables() : void
    {
        $params = $this->getParams();
        if(count($this->weights)!==count($params)) {
            throw new LogicException('Weights are not allocated: '.$this->basename($this));
        }
        foreach(array_map(null,$this->weights,$params) as [$variable,$param]) {
            if($param!==null) {
                $variable->assign($param, reference: true);
                $variable->setName('weights:'.$this->basename($this));
            }
        }
        $this->assignedWeights = true;
    }

    public function variables() : array
    {
        return $this->weights();
    }

    public function trainableVariables() : array
    {
        return array_filter($this->weights(),fn($v)=>$v->isTrainable());
    }

    public function submodules() : array
    {
        return [];
    }

    protected function basename($object) : string
    {
        $classname = get_class($object);
        return substr($classname,strrpos($classname,'\\')+1);
    }

    protected function typename($object) : string
    {
        if(is_object($object)) {
            return get_class($object);
        } else {
            return gettype($object);
        }
    }

    public function isAwareOf(string $name) : bool
    {
        return isset($this->callOptions);
    }
}
