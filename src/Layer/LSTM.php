<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class LSTM extends AbstractRNNLayer
{
    use GenericUtils;
    protected $backend;
    protected $units;
    protected $activationName;
    protected $recurrentActivationName;
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

    public function __construct(
        object $backend,
        int $units,
        array $input_shape=null,
        string|object $activation=null,
        string|object $recurrent_activation=null,
        bool $use_bias=null,
        string|callable $kernel_initializer=null,
        string|callable $recurrent_initializer=null,
        string|callable $bias_initializer=null,
        bool $return_sequences=null,
        bool $return_state=null,
        bool $go_backwards=null,
        bool $stateful=null,
        string $name=null,
        )
    {
        // defaults
        $input_shape = $input_shape ?? null;
        $activation = $activation ?? 'tanh';
        $recurrent_activation = $recurrent_activation ?? 'sigmoid';
        $use_bias = $use_bias ?? true;
        $kernel_initializer = $kernel_initializer ?? 'glorot_uniform';
        $recurrent_initializer = $recurrent_initializer ?? 'orthogonal';
        $bias_initializer = $bias_initializer ?? 'zeros';
        $return_sequences = $return_sequences ?? false;
        $return_state = $return_state ?? false;
        $go_backwards = $go_backwards ?? false;
        $stateful = $stateful ?? false;
        $name = $name ?? null;
        //'kernel_regularizer'=>null, 'bias_regularizer'=>null,
        //'activity_regularizer'=null,
        //'kernel_constraint'=null, 'bias_constraint'=null,
        
        $this->backend = $K = $backend;
        $this->activationName = $activation;
        $this->recurrentActivationName = $recurrent_activation;
        $this->units = $units;
        $this->inputShape = $input_shape;
        if($use_bias) {
            $this->useBias = $use_bias;
        }
        $this->allocateWeights($this->useBias?3:2);
        $this->kernelInitializerName = $kernel_initializer;
        $this->recurrentInitializerName = $recurrent_initializer;
        $this->biasInitializerName = $bias_initializer;
        $this->returnSequences=$return_sequences;
        $this->returnState = $return_state;
        $this->goBackwards = $go_backwards;
        $this->stateful = $stateful;
        $this->initName($name,'lstm');
        $this->cell = new LSTMCell(
            $this->backend,
            $this->units,
            activation:$activation,
            recurrent_activation:$recurrent_activation,
            use_bias:$this->useBias,
            kernel_initializer:$this->kernelInitializerName,
            recurrent_initializer:$this->recurrentInitializerName,
            bias_initializer:$this->biasInitializerName,
            );
    }

    public function build($variables=null, array $sampleWeights=null)
    {
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
        $this->cell->build([$this->feature], sampleWeights:$sampleWeights);
        $this->statesShapes = [
            [$this->units],
            [$this->units],
        ];
        if($this->returnSequences){
            $this->outputShape = [$this->timesteps,$this->units];
        }else{
            $this->outputShape = [$this->units];
        }
        $this->syncWeightVariables();
    }

    public function getConfig() : array
    {
        return [
            'units' => $this->units,
            'options' => [
                'input_shape'=>$this->inputShape,
                'use_bias'=>$this->useBias,
                'activation'=>$this->activationName,
                'recurrent_activation'=>$this->recurrentActivationName,
                'kernel_initializer' => $this->kernelInitializerName,
                'recurrent_initializer' => $this->recurrentInitializerName,
                'bias_initializer' => $this->biasInitializerName,
                'return_sequences'=>$this->returnSequences,
                'return_state'=>$this->returnState,
                'go_backwards'=>$this->goBackwards,
                'stateful'=>$this->stateful,
            ]
        ];
    }

    protected function numOfOutputStates($options)
    {
        if($this->returnState)
            return 2;
        return 0;
    }
}
