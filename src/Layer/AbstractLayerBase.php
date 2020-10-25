<?php
namespace Rindow\NeuralNetworks\Layer;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use Rindow\NeuralNetworks\Activation\FunctionFactory;
use Rindow\NeuralNetworks\Activation\Activation as ActivationInterface;

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

    public function build(array $inputShape=null, array $options=null) : array
    {
        if($inputShape!==null)
            $this->inputShape = $inputShape;
        $this->outputShape = $inputShape;
        return $this->outputShape;
    }

    protected function normalizeInputShape(array $inputShape=null) : array
    {
        if($inputShape===null)
            $inputShape = $this->inputShape;
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
                throw new InvalidArgumentException('unmatch shape of state: '.$shape.', must be '.$outputShape.' in '.$this->name.':'.$direction);
            }
        }
    }

    protected function registerLayer(LayerBase $layer,array $inputShape=null) : array
    {
        $this->layers[] = $layer;
        $outputShape = $layer->build($inputShape);
        $name = basename(str_replace('\\',DIRECTORY_SEPARATOR,get_class($layer)));
        $layer->setName($name);
        $this->addWeights($layer);
        return $outputShape;
    }
}
