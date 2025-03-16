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
use Rindow\NeuralNetworks\Gradient\MaskedNDArray;
use Rindow\NeuralNetworks\Gradient\Core\MaskedNDArray as MaskedNDArrayImpl;

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
    protected ?int $inputDtype=null;

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

    public function build(mixed $variable=null, ?array $sampleWeights=null) : void
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
    protected function normalizeInputShape(array|Variable|null $variable=null) : array
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
    protected function normalizeInputShapes(?array $variables=null) : array
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

    public function inputDtype() : ?int
    {
        return $this->inputDtype;
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
        if(count($this->weights)!=0) {
            throw new LogicException("reverseSyncWeightVariables is not impremented in ".get_class($this));
        }
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
        $string = '(';
        foreach($shape as $value) {
            if($string!='(') {
                $string .= ',';
            }
            $string .= $this->shapeToString($value);
        }
        $string .= ')';
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

    /**
     * @param array<string> $names
     */
    protected function allocateWeights(array $names, ?int $nonTrainables=null) : void
    {
        $variables = [];
        foreach($names as $name) {
            $fullname = $name.'@'.$this->name();
            $variables[] = new Variable(
                $this->backend,
                null,
                name:$fullname,
                undetermined: true,
            );
        }
        if($nonTrainables) {
            for($i=0;$i<$nonTrainables;$i++) {
                $variables[] = new Variable(
                    $this->backend,
                    null,
                    name:('nonTrainable'.$i.'@'.$this->name()),
                    undetermined: true,
                    trainable: false,
                );
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

    /**
     * @param array<NDArray>|NDArray $inputs
     * @param array<NDArray|null>|NDArray $previousMask
     * @return array<NDArray>|NDArray|null
     */
    public function computeMask(
        array|NDArray $inputs,
        array|NDArray|null $previousMask
        ) : array|NDArray|null
    {
        return null;
    }

    public function retrieveSingleMask(NDArray $input) : ?NDArray
    {
        $prevMask = null;
        if($input instanceof MaskedNDArray) {
            $prevMask = $input->mask();
        }
        return $prevMask;
    }

    protected function maskedValue(NDArray $value, NDArray $mask) : MaskedNDArray
    {
        return new MaskedNDArrayImpl($value,$mask);
    }

    public function makeSingleMaskedValue(NDArray $input, NDArray $output) : NDArray
    {
        $prevMask = null;
        if($input instanceof MaskedNDArray) {
            $prevMask = $input->mask();
        }
        $mask = $this->computeMask($input,$prevMask);
        if($mask!=null) {
            $output = $this->maskedValue($output,$mask);
        }
        return $output;
    }

    /**
     * @param array<NDArray> $inputs
     * @return array<NDArray|null>
     */
    public function retrieveMultiMasks(array $inputs) : array
    {
        $prevMasks = [];
        foreach ($inputs as $input) {
            if($input instanceof MaskedNDArray) {
                $prevMasks[] = $input->mask();
            } else {
                $prevMasks[] = null;
            }
        }
        return $prevMasks;
    }

    /**
     * @param array<NDArray> $inputs
     * @param array<NDarray> $outputs
     * @return array<NDArray>
     */
    public function makeMultiMaskedValues(array $inputs, array $outputs) : array
    {
        $prevMasks = array_map(
            fn($in)=>is_a($in,MaskedNDArray::class)?$in->mask():null,
            $inputs
        );
        $masks = $this->computeMask($inputs,$prevMasks);
        if($masks==null) {
            $values = $outputs;
        } else {
            $values = array_map(
                fn($out,$mask)=>($mask==null)?$out:$this->maskedValue($out,$mask),
                $outputs,$masks
            );
        }
        return $values;
    }
}
