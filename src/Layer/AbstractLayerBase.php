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
    protected object $backend;
    /** @var array<int|array<int>> $inputShape */
    protected ?array $inputShape=null;
    /** @var array<int> $outputShape */
    protected array $outputShape;
    protected ?ActivationInterface $activation=null;
    protected ?string $activationName=null;
    /** @var array<NDArray> $params */
    protected array $params=[];
    /** @var array<NDArray> $grads */
    protected array $grads=[];
    protected bool $shapeInspection = true;
    protected ?string $name;
    // dynamic step interfaces
    //protected $inputsVariables;
    //protected $outputsVariables;
    //protected $generation;
    /** @var array<Variable> $weights */
    protected array $weights=[];
    protected bool $assignedWeights = false;
    protected bool $built = false;
    protected bool $training = false;
    /** @var array<string,bool> $callOptions */
    protected array $callOptions = [];

    public function __construct(object $backend)
    {
        $this->backend = $backend;
    }

    public function isBuilt() : bool
    {
        return $this->built;
    }

    public function getActivation() : ?ActivationInterface
    {
        return $this->activation;
    }

    protected function setActivation(null|string|ActivationInterface $activation) : void
    {
        $this->activation = $this->createFunction($activation);
        $this->activationName = $this->toStringName($activation);
    }

    protected function createFunction(
        null|string|ActivationInterface $activation=null) : ?ActivationInterface
    {
        if($activation==null){
            return null;
        } elseif(is_string($activation)) {
            // for compiling lossfunction
            return FunctionFactory::factory($this->backend,$activation);
        } elseif($activation instanceof ActivationInterface) {
            return $activation;
        } else {
            throw new InvalidArgumentException('activation function must have the Activation interface');
        }
    }

    public function build(mixed $variable=null, array $sampleWeights=null) : void
    {
        $inputShape = $this->normalizeInputShape($variable);
        if($inputShape!==null)
            $this->inputShape = $inputShape;
        $this->outputShape = $inputShape;
    }

    /**
     * @param array<int>|Variable $variable
     * @return array<int>
     */
    protected function normalizeInputShape(array|Variable $variable=null) : array
    {
        $inputShape = null;
        if($variable===null) {
            $inputShape = $this->inputShape;
        } elseif($variable instanceof Variable) {
            $inputShape = $variable->valueShape();
            if($inputShape===null) {
                $inputShape = $this->inputShape;
            }
        } elseif(is_array($variable)) {
            $inputShape = [];
            foreach($variable as $v) {
                if(!is_int($v)) {
                    throw new InvalidArgumentException('variable must Variable or shape');
                }
                $inputShape[] = $v;
            }
        } else {
            throw new InvalidArgumentException('variable must be Variable type or null. "'.$this->typename($variable).'" given.');
        }

        if($inputShape===null) {
            throw new InvalidArgumentException("inputShape must be spacified");
        }
        return $this->fixInputShape($inputShape);
    }

    /**
     * @param array<int> $inputShape
     * @return array<int>
     */
    protected function normalizeCellInputShape(?array $inputShape) : array
    {
        if($inputShape==null) {
            $inputShape = $this->inputShape;
        }
        return $this->fixInputShape($inputShape);
    }

    /**
     * @param array<int> $inputShape
     * @return array<int>
     */
    protected function fixInputShape(?array $inputShape) : array
    {
        if($this->inputShape===null) {
            $this->inputShape = $inputShape;
        }
        if($this->shapeInspection && $this->inputShape!==$inputShape) {
            $msg = 'Input shape is inconsistent: defined as '.$this->shapeToString($this->inputShape).
            ' but '.$this->shapeToString($inputShape).' given in '.$this->basename($this);
            throw new InvalidArgumentException($msg);
        } elseif($this->inputShape===null && $inputShape===null) {
            throw new InvalidArgumentException('Input shape is not defined');
        }
        //$this->inputShape = $inputShape;
        return $inputShape;
    }

    /**
     * @param array<Variable> $variables
     * @return array<array<int>>
     */
    protected function normalizeInputShapes(array $variables=null) : array
    {
        if($variables===null) {
            $inputShapes = $this->inputShape;
        } else {
            $inputShapes = [];
            foreach($variables as $idx => $v) {
                if(!($v instanceof Variable)) {
                    throw new InvalidArgumentException('variable list must contain Variables: "'.$this->typename($v).'" included in #'.$idx.'.');
                }
                $shape = $v->valueShape();
                if(!is_array($shape)) {
                    throw new InvalidArgumentException('Variable list include unspecified shape value.'.' in #'.$idx.'.');
                }
                $inputShapes[] = $shape;
            }
        }

        if($inputShapes===null) {
            throw new InvalidArgumentException("inputShape must be spacified");
        }
        return $this->fixInputShapes($inputShapes);
    }

    /**
     * @param array<array<int>> $inputShapes
     * @return array<array<int>>
     */
    protected function fixInputShapes(?array $inputShapes) : array
    {
        if($this->inputShape===null) {
            $this->inputShape = $inputShapes;
        }
        if($this->shapeInspection && $this->inputShape!==$inputShapes) {
            if(is_array($this->inputShape)) {
                $msg = 'Input shape is inconsistent: defined as '.$this->shapeToString($this->inputShape).
                ' but '.$this->shapeToString($inputShapes).' given in '.$this->basename($this);
            } else {
                $msg = 'Input shape is inconsistent';
            }
            throw new InvalidArgumentException($msg);
        } elseif($this->inputShape===null && $inputShapes===null) {
            throw new InvalidArgumentException('Input shape is not defined');
        }
        //$this->inputShape = $inputShape;
        return $inputShapes;
    }

    public function inputShape() : array
    {
        return $this->inputShape;
    }

    public function outputShape() : array
    {
        return $this->outputShape;
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

    public function name() : string
    {
        return $this->name;
    }

    public function setShapeInspection(bool $enable) : void
    {
        $this->shapeInspection = $enable;
    }

    protected function shapeToString(mixed $shape) : string
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

    protected function toStringName(null|object|string $name) : ?string
    {
        if($name===null||is_string($name)) {
            return $name;
        }
        return get_class($name);
    }

    protected function assertInputShape(NDArray $inputs,string $direction) : void
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

    protected function assertOutputShape(NDArray $outputs,string $direction) : void
    {
        if(!$this->shapeInspection) {
            return;
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

    protected function allocateWeights(int $num,int $nonTrainables=null) : void
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

    protected function basename(object $object) : string
    {
        $classname = get_class($object);
        return substr($classname,strrpos($classname,'\\')+1);
    }

    protected function typename(mixed $object) : string
    {
        if(is_object($object)) {
            return get_class($object);
        } else {
            return gettype($object);
        }
    }

    public function isAwareOf(string $name) : bool
    {
        return isset($this->callOptions[$name]);
    }
}
