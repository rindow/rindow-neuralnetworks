<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;

class SimpleRNNCell extends AbstractRNNCell
{
    protected $backend;
    protected $units;
    protected $activation;
    protected $useBias;
    protected $kernelInitializer;
    protected $recurrentInitializer;
    protected $biasInitializer;
    protected $kernelInitializerName;
    protected $recurrentInitializerName;
    protected $biasInitializerName;

    protected $kernel;
    protected $recurrentKernel;
    protected $bias;
    protected $dKernel;
    protected $dRecurrentKernel;
    protected $dBias;
    protected $inputs;

    public function __construct(
        object $backend,
        int $units,
        array $input_shape=null,
        string|object $activation=null,
        bool $use_bias=null,
        string|callable $kernel_initializer=null,
        string|callable $recurrent_initializer=null,
        string|callable $bias_initializer=null,
    )
    {
        // defaults
        $input_shape = $input_shape ?? null;
        $activation = $activation ?? 'tanh';
        $use_bias = $use_bias ?? true;
        $kernel_initializer = $kernel_initializer ?? 'glorot_uniform';
        $recurrent_initializer = $recurrent_initializer ?? 'orthogonal';
        $bias_initializer = $bias_initializer ?? 'zeros';
        //'kernel_regularizer'=>null, 'bias_regularizer'=>null,
        //'activity_regularizer'=null,
        //'kernel_constraint'=null, 'bias_constraint'=null,
        
        $this->backend = $K = $backend;
        $this->units = $units;
        $this->inputShape = $input_shape;

        if($use_bias) {
            $this->useBias = $use_bias;
        }
        $this->setActivation($activation);
        $this->kernelInitializer = $K->getInitializer($kernel_initializer);
        $this->recurrentInitializer = $K->getInitializer($recurrent_initializer);
        $this->biasInitializer   = $K->getInitializer($bias_initializer);
        $this->kernelInitializerName = $kernel_initializer;
        $this->recurrentInitializerName = $recurrent_initializer;
        $this->biasInitializerName = $bias_initializer;
    }

    public function build($inputShape=null, array $sampleWeights=null)
    {
        $K = $this->backend;
        $kernelInitializer = $this->kernelInitializer;
        $recurrentInitializer = $this->recurrentInitializer;
        $biasInitializer = $this->biasInitializer;

        $inputShape = $this->normalizeCellInputShape($inputShape);
        //if(count($inputShape)!=1) {
        //    throw new InvalidArgumentException(
        ///        'Unsuppored input shape: ['.implode(',',$inputShape).']');
        //}
        $shape = $inputShape;
        $inputDim = array_pop($shape);
        if($this->kernel===null) {
            if($sampleWeights) {
                $this->kernel = $sampleWeights[0];
                $this->recurrentKernel = $sampleWeights[1];
                $this->bias = $sampleWeights[2];
            } else {
                $this->kernel = $kernelInitializer(
                    [$inputDim,$this->units],
                    [$inputDim,$this->units]);
                $this->recurrentKernel = $recurrentInitializer(
                    [$this->units,$this->units],
                    [$this->units,$this->units]);
                if($this->useBias) {
                    $this->bias = $biasInitializer([$this->units]);
                }
            }
        }
        $this->dKernel = $K->zerosLike($this->kernel);
        $this->dRecurrentKernel = $K->zerosLike($this->recurrentKernel);
        if($this->useBias) {
            $this->dBias = $K->zerosLike($this->bias);
        }
        array_push($shape,$this->units);
        $this->outputShape = $shape;
        return $this->outputShape;
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
            ]
        ];
    }

    protected function call(NDArray $inputs, array $states, bool $training, object $calcState, array $options=null) : array
    {
        $K = $this->backend;
        $prev_h = $states[0];
        if($this->bias){
            $outputs = $K->batch_gemm($inputs, $this->kernel,1.0,1.0,$this->bias);
        } else {
            $outputs = $K->gemm($inputs, $this->kernel);
        }
        $outputs = $K->gemm($prev_h, $this->recurrentKernel,1.0,1.0,$outputs);
        if($this->activation) {
            $calcState->activation = new \stdClass();
            $outputs = $this->activation->forward($calcState->activation,$outputs,$training);
        }

        $calcState->inputs = $inputs;
        $calcState->prev_h = $prev_h;
        return [$outputs,[$outputs]];
    }

    protected function differentiate(NDArray $dOutputs, array $dStates, object $calcState) : array
    {
        $K = $this->backend;
        $dNext_h = $dStates[0];
        $dOutputs = $K->add($dOutputs,$dNext_h);
        if($this->activation) {
            $dOutputs = $this->activation->backward($calcState->activation,$dOutputs);
        }
        $dInputs = $K->zerosLike($calcState->inputs);
        if($this->bias) {
            $K->update_add($this->dBias,$K->sum($dOutputs,$axis=0));
        }
        // Add RecurrentKernel grad
        $K->gemm($calcState->prev_h, $dOutputs,1.0,1.0,$this->dRecurrentKernel,true);
        // backward PrevOutput grad
        $dPrev_h = $K->gemm($dOutputs, $this->recurrentKernel,1.0,0,null,false,true);
        // Add Kernel grad
        $K->gemm($calcState->inputs, $dOutputs,1.0,1.0,$this->dKernel,true);
        // backward inputs grad
        $dInputs = $K->gemm($dOutputs, $this->kernel,1.0,0,null,false,true);
        return [$dInputs, [$dPrev_h]];
    }
}
