<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class GRU extends AbstractRNNLayer
{
    use GenericUtils;
    protected ?string $recurrentActivationName;
    protected bool $useBias;
    protected bool $resetAfter;
    //protected $timesteps;
    //protected $feature;

    /**
     * @param array<int> $input_shape
     */
    public function __construct(
        object $backend,
        int $units,
        ?array $input_shape=null,
        string|object|null $activation=null,
        string|object|null $recurrent_activation=null,
        ?bool $use_bias=null,
        string|callable|null $kernel_initializer=null,
        string|callable|null $recurrent_initializer=null,
        string|callable|null $bias_initializer=null,
        ?bool $return_sequences=null,
        ?bool $return_state=null,
        ?bool $go_backwards=null,
        ?bool $stateful=null,
        ?bool $reset_after=null,
        ?string $name=null,
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
        $reset_after = $reset_after ?? true;
        $name = $name ?? null;
        //'kernel_regularizer'=>null, 'bias_regularizer'=>null,
        //'activity_regularizer'=null,
        //'kernel_constraint'=null, 'bias_constraint'=null,
        
        parent::__construct($backend);
        $K = $backend;
        $this->activationName = $this->toStringName($activation);
        $this->recurrentActivationName = $this->toStringName($recurrent_activation);
        $this->setUnits($units);
        $this->inputShape = $input_shape;
        $this->useBias = $use_bias;
        $this->initName($name,'gru');
        $this->allocateWeights(
            $this->useBias?
            ['kernel','recurrentKernel','bias']:
            ['kernel','recurrentKernel']
        );
        $this->setKernelInitializerNames(
            $kernel_initializer,
            $recurrent_initializer,
            $bias_initializer,
        );
        $this->setFlags(
            returnSequences:$return_sequences,
            returnState:$return_state,
            goBackwards:$go_backwards,
            stateful:$stateful,
        );
        $this->resetAfter = $reset_after;
        $this->setCell(new GRUCell(
            $this->backend,
            $this->units,
            activation:$activation,
            recurrent_activation:$recurrent_activation,
            use_bias:$this->useBias,
            kernel_initializer:$this->kernelInitializerName,
            recurrent_initializer:$this->recurrentInitializerName,
            bias_initializer:$this->biasInitializerName,
            reset_after:$this->resetAfter,
        ));
    }

    public function build(mixed $variables=null, ?array $sampleWeights=null) : void
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
        $timesteps = $inputShape[0];
        $feature = $inputShape[1];
        $this->cell()->build([$feature], sampleWeights:$sampleWeights);
        $this->statesShapes = [
            [$this->units],
        ];

        if($this->returnSequences){
            $this->outputShape = [$timesteps,$this->units];
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
                'reset_after'=>$this->resetAfter,
            ]
        ];
    }

    //protected function numOfOutputStates($options) : int
    //{
    //    if($this->returnState)
    //        return 1;
    //    return 0;
    //}
}
