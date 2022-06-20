<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class SimpleRNN extends AbstractRNNLayer
{
    use GenericUtils;
    protected $backend;
    protected $units;
    protected $activation;
    protected $useBias;
    protected $kernelInitializerName;
    protected $recurrentInitializerName;
    protected $biasInitializerName;
    protected $returnSequences;
    protected $returnState;
    protected $goBackwards;
    protected $stateful;
    protected $cell;
    protected $timesteps;
    protected $feature;

    protected $calcStates;
    protected $initialStates;
    protected $origInputsShape;

    public function __construct($backend,int $units, array $options=null)
    {
        extract($this->extractArgs([
            'input_shape'=>null,
            'activation'=>'tanh',
            'use_bias'=>true,
            'kernel_initializer'=>'glorot_uniform',
            'recurrent_initializer'=>'orthogonal',
            'bias_initializer'=>'zeros',
            'return_sequences'=>false,
            'return_state'=>false,
            'go_backwards'=>false,
            'stateful'=>false,
            //'kernel_regularizer'=>null, 'bias_regularizer'=>null,
            //'activity_regularizer'=null,
            //'kernel_constraint'=null, 'bias_constraint'=null,
        ],$options));
        $this->backend = $K = $backend;
        $this->units = $units;
        $this->inputShape = $input_shape;
        if($use_bias) {
            $this->useBias = $use_bias;
        }
        $this->setActivation($activation);
        $activation = $this->activation;
        $this->activation = null;
        $this->kernelInitializerName = $kernel_initializer;
        $this->recurrentInitializerName = $recurrent_initializer;
        $this->biasInitializerName = $bias_initializer;
        $this->returnSequences=$return_sequences;
        $this->returnState = $return_state;
        $this->goBackwards = $go_backwards;
        $this->stateful = $stateful;
        $this->cell = new SimpleRNNCell(
            $this->backend,
            $this->units,
            [
            'activation'=>$activation,
            'use_bias'=>$this->useBias,
            'kernel_initializer'=>$this->kernelInitializerName,
            'recurrent_initializer'=>$this->recurrentInitializerName,
            'bias_initializer'=>$this->biasInitializerName,
            ]);
    }

    public function build($variables=null, array $options=null)
    {
        extract($this->extractArgs([
            'sampleWeights'=>null,
        ],$options));
        $K = $this->backend;

        if(is_object($variables)) {
            $variables = [$variables];
        }
        $inputShape = $this->normalizeInputShape(($variables===null)?null:$variables[0]);
        //if(count($inputShape)!=1) {
        //    throw new InvalidArgumentException(
        ///        'Unsuppored input shape: ['.implode(',',$inputShape).']');
        //}
        if(count($inputShape)!=2){
            throw new InvalidArgumentException('Unsuppored input shape.:['.implode(',',$inputShape).']');
        }
        $this->timesteps = $inputShape[0];
        $this->feature = $inputShape[1];
        $this->cell->build([$this->feature],$options);
        $this->statesShapes = [[$this->units]];
        if($this->returnSequences){
            $this->outputShape = [$this->timesteps,$this->units];
        }else{
            $this->outputShape = [$this->units];
        }
        $this->normalizeInitialStatesShape($variables,$this->statesShapes);
        if($this->returnState) {
            return $this->createOutputDefinition(array_merge([$this->outputShape],$this->statesShapes));
        } else {
            return $this->createOutputDefinition([$this->outputShape]);
        }
    }

    public function setShapeInspection(bool $enable)
    {
        parent::setShapeInspection($enable);
        $this->cell->setShapeInspection($enable);
    }

    public function getParams() : array
    {
        return $this->cell->getParams();
    }

    public function getGrads() : array
    {
        return $this->cell->getGrads();
    }

    public function getConfig() : array
    {
        return [
            'units' => $this->units,
            'options' => [
                'input_shape'=>$this->inputShape,
                'use_bias'=>$this->useBias,
                'activation'=>$this->activationName,
                'kernel_initializer' => $this->kernelInitializerName,
                'recurrent_initializer' => $this->recurrentInitializerName,
                'bias_initializer' => $this->biasInitializerName,
                'return_sequences'=>$this->returnSequences,
                'return_state'=>$this->returnState,
                'go_backward'=>$this->goBackwards,
                'stateful'=>$this->stateful,
            ]
        ];
    }

    protected function call(NDArray $inputs, bool $training, array $initialStates=null, array $options=null)
    {
        return $this->callCell($inputs,$training,$initialStates,$options);
    }

    protected function differentiate(NDArray $dOutputs, array $dStates=null)
    {
        return $this->differentiateCell($dOutputs,$dStates);
    }

    protected function numOfOutputStates($options)
    {
        if($this->returnState)
            return 1;
        return 0;
    }
}
