<?php
namespace Rindow\NeuralNetworks\Layer;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use Rindow\NeuralNetworks\Activation\FunctionFactory;
use Rindow\NeuralNetworks\Activation\Activation as ActivationInterface;
use Rindow\NeuralNetworks\Gradient\Core\Variable;
use Rindow\NeuralNetworks\Gradient\Core\Undetermined;
use Rindow\NeuralNetworks\Gradient\Core\UndeterminedNDArray;
use Rindow\NeuralNetworks\Model\BuildContext;
use Rindow\NeuralNetworks\Layer\Embedding;
/**
 *
 */
abstract class AbstractLayerBase implements LayerBase
{
    protected $layers = [];
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
    protected $inputsVariables;
    protected $outputsVariables;
    protected $generation;
    protected $weights;

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

    public function build($variable=null, array $options=null)
    {
        $inputShape = $this->normalizeInputShape($variable);
        if($inputShape!==null)
            $this->inputShape = $inputShape;
        $this->outputShape = $inputShape;
        return $this->createOutputDefinition([$this->outputShape]);
    }

    protected function normalizeInputShape($variables=null) : array
    {
        //if($variables instanceof UndeterminedNDArray) {
        //    if($variables->isNull()) {
        //        $variables = null;
        //    } else {
        //        $variables = new Undetermined($variables);
        //    }
        //}
        if($variables===null) {
            $inputShape = $this->inputShape;
            $variables = [new Undetermined()];
        } elseif($variables instanceof Variable) {
            $inputShape = $variables->valueShape();
            if($inputShape===null) {
                $inputShape = $this->inputShape;
            }
            $variables = [$variables];
        } elseif(is_array($variables)) {
            $inputShape = [];
            foreach($variables as $v) {
                //if($v instanceof UndeterminedNDArray) {
                //    $v = new Undetermined($v);
                //}
                if(!($v instanceof Variable)) {
                    throw new InvalidArgumentException('variable list must contain Variables: '.gettype($v).' included');
                }
                $inputShape[] = $v->valueShape();
            }
        } else {
            throw new InvalidArgumentException('variable must be Variable type or null');
        }
        $this->inputsVariables = $variables;
        $this->generation = array_reduce(
            $variables,function($max,$x){return max($max,$x->generation());},-1);
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
        if($this->inputShape===null)
            $this->inputShape = $inputShape;
        if($this->inputShape!==$inputShape) {
            if(is_array($this->inputShape)&&is_int($this->inputShape[0])) {
                $msg = 'Input shape is inconsistent: ['.implode(',',$this->inputShape).
                '] and ['.implode(',',$inputShape).']';
            } else {
                $msg = 'Input shape is inconsistent';
            }
            throw new InvalidArgumentException($msg);
        } elseif($inputShape===null) {
            throw new InvalidArgumentException('Input shape is not defined');
        }
        $this->inputShape = $inputShape;
        return $inputShape;
    }

    protected function normalizeInitialStatesShape(array $variables=null,$statesShapes=null) : void
    {
        if($variables===null) {
            return;
        }
        $this->inputsVariables = $variables;
        $this->generation = array_reduce(
            $variables,function($max,$x){return max($max,$x->generation());},-1);
        array_shift($variables);
        foreach($variables as $k => $v) {
            if($v instanceof Undetermined) {
                $nd = $v->value();
                $shape = $statesShapes[$k];
                array_unshift($shape,1);
                if($nd!=null) {
                    $nd->setShape($shape);
                } else {
                    $v->setValue(new UndeterminedNDArray($shape));
                }
            }
        }
    }

    protected function createOutputDefinition(array $outputShapes)
    {
        $defines = [];
        foreach ($outputShapes as $outputShape) {
            array_unshift($outputShape,1);
            $define = new Undetermined(new UndeterminedNDArray($outputShape));
            $define->setCreator($this);
            $defines[] = $define;
        }
        $this->outputsVariables = array_map(function($o){return $o->reference();},$defines);
        if(BuildContext::$build) {
            BuildContext::add($this);
        }
        if(count($defines)==1) {
            return $defines[0];
        }
        return $defines;
    }

    public function outputShape() : array
    {
        return $this->outputShape;
    }

    public function statesShapes() : array
    {
        return $this->statesShapes;
    }

    protected function addWeights($weights,$grads=null)
    {
        if($weights instanceof LayerBase){
            $this->params = array_merge($this->params,$weights->getParams());
            $this->grads  = array_merge($this->grads, $weights->getGrads());
            return;
        }elseif($weights instanceof NDArray){
            if($grads==null){
                throw new InvalidArgumentException('need grads to add weights');
            }
            $this->params[]=$weights;
            $this->grads[]=$grads;
        }else{
            throw new InvalidArgumentException('invalid type to add weights');
        }
    }

    public function getParams() : array
    {
        return $this->params;
    }

    public function getGrads() : array
    {
        return $this->grads;
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
        foreach ($this->layers as $layer) {
            $layer->setShapeInspection($enable);
        }
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
            throw new InvalidArgumentException('unmatch input shape: '.$shape.', must be '.$inputShape.' in '.$this->name.':'.$direction);
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
            throw new InvalidArgumentException('unmatch output shape: '.$shape.', must be '.$outputShape.' in '.$this->name.':'.$direction);
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
                throw new InvalidArgumentException('unmatch shape of state: '.$shape.', must be '.$stateShape.' in '.$this->name.':'.$direction);
            }
        }
    }

    protected function registerLayer(LayerBase $layer,array $inputShape=null) : array
    {
        $this->layers[] = $layer;
        $outputShape = $layer->build($inputShape);
        $name = $this->basename($layer);
        $layer->setName($name);
        $this->addWeights($layer);
        return $outputShape;
    }

    /*
    *  dynamic step interfaces
    */

    /**
    *   @return array<Variable>
    */
    public function inputs()
    {
        return $this->inputsVariables;
    }

    /**
    *   @return array<Variable>
    */
    public function outputs()
    {
        return $this->outputsVariables;
    }

    /**
    *  @return int
    */
    public function generation() : int
    {
        return $this->generation;
    }

    /**
    *  @return array<Variable>
    *       outputs
    */
    public function weights()
    {
        if($this->weights) {
            return $this->weights;
        }
        $variables = [];
        foreach (array_map(null,$this->getParams(),$this->getGrads()) as $values) {
            [$param, $grads] = $values;
            $variable = new Variable($this->backend,$param);
            $variable->setGrad($grads);
            $variable->setName('weights:'.$this->basename($this));
            $variables[] = $variable;
        }
        $this->weights = $variables;
        return $variables;
    }

    protected function basename($object) : string
    {
        $classname = get_class($object);
        return substr($classname,strrpos($classname,'\\')+1);
    }
}
