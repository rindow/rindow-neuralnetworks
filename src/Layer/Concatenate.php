<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;
use Rindow\NeuralNetworks\Gradient\Core\GradientTape;
use Rindow\NeuralNetworks\Gradient\Core\GradientUtils;
use Rindow\NeuralNetworks\Gradient\Core\Variable;
use Rindow\NeuralNetworks\Gradient\Core\Undetermined;
use Rindow\NeuralNetworks\Gradient\Core\UndeterminedNDArray;
use Rindow\NeuralNetworks\Model\BuildContext;

class Concatenate extends AbstractLayerBase
{
    use GenericUtils;
    use GradientUtils;
    protected $backend;
    protected $axis;
    protected $shapes;

    public function __construct(
        $backend,
        array $options=null)
    {
        extract($this->extractArgs([
            'axis'=>-1,
            'input_shapes'=>null,
        ],$options));
        $this->backend = $backend;
        if(!is_int($axis)) {
            throw new InvalidArgumentException('axis must be integer.');
        }
        $this->axis = $axis;
        $this->inputShape = $input_shapes;
    }

    public function build($variables=null, array $options=null)
    {
        $K = $this->backend;
        if(!is_array($variables) && $variables!==null) {
            throw new InvalidArgumentException('inputs must be list of variable');
        }
        $inputShapes = $this->normalizeInputShape($variables);
        if(count($inputShapes)<2) {
            throw new InvalidArgumentException('num of inputs must be greater then 2 or equal: input dims is '.count($inputShape));
        }
        $m = 0;
        $baseShape = null;
        foreach ($inputShapes as $idx => $shape) {
            if(!is_array($shape)) {
                throw new InvalidArgumentException('input_shapes must be the list of shape:');
            }
            if($this->axis < 0) {
                $axis = 1 + count($shape) + $this->axis;
            } else {
                $axis = $this->axis;
            }
            $shapestack = [];
            // skip batchsize and axis
            for($i=1;$i<$axis;$i++) {
                $shapestack[] = array_shift($shape);
            }
            $m += array_shift($shape);
            if($baseShape===null) {
                $baseShape = array_merge($shapestack,$shape);
            } else {
                if($baseShape != array_merge($shapestack,$shape)) {
                    $msg = '';
                    foreach($inputShapes as $shape) {
                        $msg .= '['.implode(',',$shape).'] ';
                    }
                    throw new InvalidArgumentException('Unmatch input_shapes:'.$msg);
                }
            }
        }
        array_unshift($shape,$m);
        $shape = array_merge($shapestack,$shape);
        $this->outputShape = $shape;
        return $this->createOutputDefinition([$this->outputShape]);
    }

    public function getParams() : array
    {
        return [];
    }

    public function getGrads() : array
    {
        return [];
    }

    public function getConfig() : array
    {
        return [
            'options' => [
                'axis'=>$this->axis,
                'input_shapes'=>$this->inputShape,
            ]
        ];
    }

    protected function assertInputShapes(array $inputs,$direction)
    {
        if(!$this->shapeInspection)
            return;
        if($this->inputShape===null) {
            throw new InvalidArgumentException('Uninitialized input shape');
        }
        if(count($inputs)!=count($this->inputShape)){
            throw new InvalidArgumentException('Unmatch num of input. inputs need '.count($this->inputShapes).' NDArray. '.count($inputs).'given in '.$this->name.':'.$direction);
        }
        $batchSize = null;
        foreach($inputs as $idx=>$input){;
            $inputShape = $this->inputShape[$idx];
            $shape = $input->shape();
            if($batchSize === null) {
                $batchSize = array_shift($shape);
            } else {
                if($batchSize != array_shift($shape)) {
                    throw new InvalidArgumentException('unmatch batch size of input '.$idx.': ['.$batchSize.'] in '.$this->name.':'.$direction);
                }
            }
            if($shape!=$inputShape) {
                $shape = $shape ? implode(',',$shape) : '';
                throw new InvalidArgumentException('unmatch shape of input '.$idx.': ['.$shape.'], must be ['.implode(',',$inputShape).'] in '.$this->name.':'.$direction);
            }
        }
    }

    public function forward(array $inputs, bool $training)
    {
        if(BuildContext::$build) {
            return $this->build($inputs);
        }
        if(count($inputs)<2) {
            throw new InvalidArgumentException('Must have arguments greater than 2 or equal');
        }
        $this->assertInputShapes($inputs,'forward');
        $outputs = $this->call($inputs,$training);
        $this->assertOutputShape($outputs,'forward');
        return $outputs;
    }

    public function backward(array $dOutputs) : array
    {
        if(count($dOutputs)!=1) {
            throw new InvalidArgumentException('dOutputs must be list containing one NDArray');
        }
        $dOutputs = $dOutputs[0];
        if(!($dOutputs instanceof NDArray)) {
            throw new InvalidArgumentException('dOutputs must be list containing one NDArray');
        }
        $this->assertOutputShape($dOutputs,'backward');
        $dInputs = $this->differentiate($dOutputs);
        $this->assertInputShapes($dInputs,'backward');
        return $dInputs;
    }

    protected function call(array $inputs, bool $training) : NDArray
    {
        $K = $this->backend;
        $outputs = $K->concat($inputs,$this->axis);
        $this->shapes = [];
        foreach ($inputs as $v) {
            $this->shapes[] = $v->shape();
        }
        return $outputs;
    }

    protected function differentiate(NDArray $dOutputs) : array
    {
        $K = $this->backend;
        if($this->axis<0) {
            $axis = count($this->shapes[0])+$this->axis;
        } else {
            $axis = $this->axis;
        }
        foreach ($this->shapes as $shape) {
            $sizeSplits[] = $shape[$axis];
        }
        $dInputs = $K->split($dOutputs,$sizeSplits,$axis);
        return $dInputs;
    }

    /**
    *  @param array<Variable>  $inputs
    *  @return array<Variable>
    */
    public function __invoke($inputs, bool $training)
    {
        if(!is_array($inputs)) {
            throw new InvalidArgumentException('inputs must be list of Variable');
        }
        $outputs = null;
        if($this->outputShape==null) {
            $outputs = $this->build($inputs);
        }
        if($inputs[0] instanceof Undetermined) {
            if($outputs===null) {
                throw new InvalidArgumentException('Undetermined is found in second calling.');
            }
            return $outputs;
        }
        $rawInputs = array_map(function($value){return $value->value();},$inputs);
        $outputs = $this->forward($rawInputs,$training);
        $outputs = $this->postGradientProcess(
            $this->backend, $inputs, [$outputs]);
        return $outputs[0];
    }
}
