<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class LSTMCell extends AbstractRNNCell
{
    use GenericUtils;
    protected $backend;
    protected $units;
    protected $useBias;
    protected $kernelInitializer;
    protected $biasInitializer;
    protected $ac;
    protected $ac_i;
    protected $ac_f;
    protected $ac_c;
    protected $ac_o;

    protected $kernel;
    protected $recurrentKernel;
    protected $bias;
    protected $dKernel;
    protected $dRecurrentKernel;
    protected $dBias;
    protected $inputs;

    public function __construct($backend,int $units, array $options=null)
    {
        extract($this->extractArgs([
            'input_shape'=>null,
            'activation'=>'tanh',
            'recurrent_activation'=>'sigmoid',
            'use_bias'=>true,
            'kernel_initializer'=>'glorot_uniform',
            'recurrent_initializer'=>'orthogonal',
            'bias_initializer'=>'zeros',
            'unit_forget_bias'=>true,
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
        $this->activation = $this->createFunction($activation);
        $this->recurrentActivation = $this->createFunction($recurrent_activation);
        $this->kernelInitializer = $K->getInitializer($kernel_initializer);
        $this->recurrentInitializer = $K->getInitializer($recurrent_initializer);
        $this->biasInitializer   = $K->getInitializer($bias_initializer);
        $this->kernelInitializerName = $kernel_initializer;
        $this->recurrentInitializerName = $recurrent_initializer;
        $this->biasInitializerName = $bias_initializer;
    }

    public function build($inputShape=null, array $options=null)
    {
        extract($this->extractArgs([
            'sampleWeights'=>null,
        ],$options));
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
        if($sampleWeights) {
            $this->kernel = $sampleWeights[0];
            $this->recurrentKernel = $sampleWeights[1];
            $this->bias = $sampleWeights[2];
        } else {
            $this->kernel = $kernelInitializer([
                $inputDim,$this->units*4],
                [$inputDim,$this->units]);
            $this->recurrentKernel = $recurrentInitializer(
                [$this->units,$this->units*4],
                [$this->units,$this->units]);
            if($this->useBias) {
                $this->bias = $biasInitializer([$this->units*4]);
            }
        }
        $this->dKernel = $K->zerosLike($this->kernel);
        $this->dRecurrentKernel = $K->zerosLike($this->recurrentKernel);
        if($this->bias) {
            $this->dBias = $K->zerosLike($this->bias);
        }
        array_push($shape,$this->units);
        $this->outputShape = $shape;
        return $this->outputShape;
    }

    public function getParams() : array
    {
        if($this->bias) {
            return [$this->kernel,$this->recurrentKernel,$this->bias];
        } else {
            return [$this->kernel,$this->recurrentKernel];
        }
    }

    public function getGrads() : array
    {
        if($this->bias) {
            return [$this->dKernel,$this->dRecurrentKernel,$this->dBias];
        } else {
            return [$this->dKernel,$this->dRecurrentKernel];
        }
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
            ]
        ];
    }

    protected function call(NDArray $inputs, array $states, bool $training, object $calcState, array $options=null) : array
    {
        $K = $this->backend;
        $prev_h = $states[0];
        $prev_c = $states[1];

        if($this->bias){
            $outputs = $K->batch_gemm($inputs, $this->kernel,1.0,1.0,$this->bias);
        } else {
            $outputs = $K->gemm($inputs, $this->kernel);
        }
        $outputs = $K->gemm($prev_h, $this->recurrentKernel,1.0,1.0,$outputs);

        $x_i = $K->slice($outputs,
            [0,0],[-1,$this->units]);
        $x_f = $K->slice($outputs,
            [0,$this->units],[-1,$this->units]);
        $x_c = $K->slice($outputs,
            [0,$this->units*2],[-1,$this->units]);
        $x_o = $K->slice($outputs,
            [0,$this->units*3],[-1,$this->units]);

        if($this->activation){
            $x_c = $this->activation->forward($x_c,$training);
            $calcState->ac_c = $this->activation->getStates();
        }
        if($this->recurrentActivation){
            $x_i = $this->recurrentActivation->forward($x_i,$training);
            $calcState->ac_i = $this->recurrentActivation->getStates();
            $x_f = $this->recurrentActivation->forward($x_f,$training);
            $calcState->ac_f = $this->recurrentActivation->getStates();
            $x_o = $this->recurrentActivation->forward($x_o,$training);
            $calcState->ac_o = $this->recurrentActivation->getStates();
        }
        $next_c = $K->add($K->mul($x_f,$prev_c),$K->mul($x_i,$x_c));
        $ac_next_c = $next_c;
        if($this->activation){
            $ac_next_c = $this->activation->forward($ac_next_c,$training);
            $calcState->ac = $this->activation->getStates();
        }
        // next_h = o * ac_next_c
        $next_h = $K->mul($x_o,$ac_next_c);

        $calcState->inputs = $inputs;
        $calcState->prev_h = $prev_h;
        $calcState->prev_c = $prev_c;
        $calcState->x_i = $x_i;
        $calcState->x_f = $x_f;
        $calcState->x_c = $x_c;
        $calcState->x_o = $x_o;
        $calcState->ac_next_c = $ac_next_c;

        return [$next_h,[$next_h,$next_c]];
    }

    protected function differentiate(NDArray $dOutputs, array $dStates, object $calcState) : array
    {
        $K = $this->backend;
        $dNext_h = $dStates[0];
        $dNext_c = $dStates[1];
        $dNext_h = $K->add($dOutputs,$dNext_h);

        $dAc_next_c = $K->mul($calcState->x_o,$dNext_h);
        if($this->activation){
            $this->activation->setStates($calcState->ac);
            $dAc_next_c = $this->activation->backward($dAc_next_c);
        }
        $dNext_c = $K->add($dNext_c, $dAc_next_c);

        $dPrev_c = $K->mul($dNext_c, $calcState->x_f);

        $dx_i = $K->mul($dNext_c,$calcState->x_c);
        $dx_f = $K->mul($dNext_c,$calcState->prev_c);
        $dx_o = $K->mul($dNext_h,$calcState->ac_next_c);
        $dx_c = $K->mul($dNext_c,$calcState->x_i);

        if($this->recurrentActivation){
            $this->recurrentActivation->setStates($calcState->ac_i);
            $dx_i = $this->recurrentActivation->backward($dx_i);
            $this->recurrentActivation->setStates($calcState->ac_f);
            $dx_f = $this->recurrentActivation->backward($dx_f);
            $this->recurrentActivation->setStates($calcState->ac_o);
            $dx_o = $this->recurrentActivation->backward($dx_o);
        }
        if($this->activation){
            $this->activation->setStates($calcState->ac_c);
            $dx_c = $this->activation->backward($dx_c);
        }

        $dOutputs = $K->stack(
            [$dx_i,$dx_f,$dx_c,$dx_o],$axis=1);
        $shape = $dOutputs->shape();
        $batches = array_shift($shape);
        $dOutputs = $dOutputs->reshape([
                $batches,
                array_product($shape)
            ]);

        $K->gemm($calcState->prev_h, $dOutputs,1.0,1.0,
            $this->dRecurrentKernel,true,false);
        $K->gemm($calcState->inputs, $dOutputs,1.0,1.0,
            $this->dKernel,true,false);
        if($this->dBias) {
            $K->update_add($this->dBias,$K->sum($dOutputs, $axis=0));
        }

        $dInputs = $K->gemm($dOutputs, $this->kernel,1.0,0.0,
            null,false,true);
        $dPrev_h = $K->gemm($dOutputs, $this->recurrentKernel,1.0,0.0,
            null,false,true);

        return [$dInputs,[$dPrev_h, $dPrev_c]];
    }
}
