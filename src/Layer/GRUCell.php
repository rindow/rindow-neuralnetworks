<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class GRUCell extends AbstractRNNCell
{
    use GenericUtils;
    protected $backend;
    protected $units;
    protected $useBias;
    protected $kernelInitializer;
    protected $biasInitializer;
    protected $resetAfter;
    protected $activation;
    protected $recurrentActivation;

    protected $kernel;
    protected $recurrentKernel;
    protected $r_kernel_z;
    protected $r_kernel_r;
    protected $r_kernel_hh;
    protected $bias;
    protected $inputBias;
    protected $recurrentBias;
    protected $dKernel;
    protected $dRecurrentKernel;
    protected $dR_kernel_z;
    protected $dR_kernel_r;
    protected $dR_kernel_hh;
    protected $dBias;
    protected $dInputBias;
    protected $dRecurrentBias;
    protected $inputs;

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
        bool $reset_after=null,
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
        $reset_after = $reset_after ?? true;
        //'kernel_regularizer'=>null, 'bias_regularizer'=>null,
        //'activity_regularizer'=null,
        //'kernel_constraint'=null, 'bias_constraint'=null,
        
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
        $this->resetAfter = $reset_after;
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
                if($this->useBias) {
                    $this->bias = $sampleWeights[2];
                }
            } else {
                $this->kernel = $kernelInitializer(
                    [$inputDim,$this->units*3],
                    [$inputDim,$this->units*3]);
                if($this->resetAfter) {
                    $this->recurrentKernel = $recurrentInitializer(
                        [$this->units,$this->units*3],
                        [$this->units,$this->units*3]);
                } else {
                    $this->recurrentKernel = $recurrentInitializer(
                        [$this->units*3,$this->units],
                        [$this->units*3,$this->units]);
                }
                if($this->useBias) {
                    if($this->resetAfter) {
                        $this->bias = $biasInitializer([2,$this->units*3]);
                    } else {
                        $this->bias = $biasInitializer([$this->units*3]);
                    }
                }
            }
        }
        $this->dKernel = $K->zerosLike($this->kernel);
        $this->dRecurrentKernel = $K->zerosLike($this->recurrentKernel);
        if($this->useBias) {
            $this->dBias = $K->zerosLike($this->bias);
            if($this->resetAfter) {
                $this->dInputBias = $this->dBias[0];
                $this->dRecurrentBias = $this->dBias[1];
            } else {
                $this->dInputBias = $this->dBias;
            }
        }
        if($this->useBias) {
            if($this->resetAfter) {
                $this->inputBias = $this->bias[0];
                $this->recurrentBias = $this->bias[1];
            } else {
                $this->inputBias = $this->bias;
            }
        }
        if(!$this->resetAfter) {
            $this->r_kernel_z = $this->recurrentKernel[[0,$this->units-1]];
            $this->r_kernel_r = $this->recurrentKernel[[$this->units,$this->units*2-1]];
            $this->r_kernel_hh = $this->recurrentKernel[[$this->units*2,$this->units*3-1]];
            $this->dR_kernel_z = $this->dRecurrentKernel[[0,$this->units-1]];
            $this->dR_kernel_r = $this->dRecurrentKernel[[$this->units,$this->units*2-1]];
            $this->dR_kernel_hh = $this->dRecurrentKernel[[$this->units*2,$this->units*3-1]];
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
                'recurrent_activation'=>$this->recurrentActivationName,
                'kernel_initializer' => $this->kernelInitializerName,
                'recurrent_initializer' => $this->recurrentInitializerName,
                'bias_initializer' => $this->biasInitializerName,
                'reset_after'=>$this->resetAfter,
            ]
        ];
    }

    protected function call(NDArray $inputs, array $states, bool $training, object $calcState, array $options=null) : array
    {
        $K = $this->backend;
        $prev_h = $states[0];

        // matrix_x = x dot w + b
        if($this->useBias){
            $matrix_x = $K->batch_gemm($inputs, $this->kernel,
                1.0,1.0,$this->inputBias);
        } else {
            $matrix_x = $K->gemm($inputs, $this->kernel);
        }

        // (z,r,hh) = matrix_x
        $x_z = $K->slice($matrix_x,
            [0,0],[-1,$this->units]);
        $x_r = $K->slice($matrix_x,
            [0,$this->units],[-1,$this->units]);
        $x_hh = $K->slice($matrix_x,
            [0,$this->units*2],[-1,$this->units]);

        if($this->resetAfter) {
            // matrix_inner = prev_h dot hw + hb
            if($this->useBias) {
                $matrix_inner = $K->batch_gemm($prev_h, $this->recurrentKernel,
                    1.0,1.0, $this->recurrentBias);
            } else {
                $matrix_inner = $K->gemm($prev_h, $this->recurrentKernel);
            }

            // (internal_z,internal_r,internal_r) = matrix_inner
            $internal_z = $K->slice($matrix_inner,
                [0,0],[-1,$this->units]);
            $internal_r = $K->slice($matrix_inner,
                [0,$this->units],[-1,$this->units]);
            $internal_hh = $K->slice($matrix_inner,
                [0,$this->units*2],[-1,$this->units]);

            // z = s(z + internal_z)
            // r = s(r + internal_r)
            $K->update_add($x_z,$internal_z);
            $K->update_add($x_r,$internal_r);
            if($this->recurrentActivation){
                $calcState->ac_z = new \stdClass();
                $x_z = $this->recurrentActivation->forward($calcState->ac_z,$x_z,$training);
                $calcState->ac_r = new \stdClass();
                $x_r = $this->recurrentActivation->forward($calcState->ac_r,$x_r,$training);
            }
            // hh = t(hh + (r * internal_hh))
            $internal_hh = $K->mul($x_r,$internal_hh);
            $K->update_add($x_hh,$internal_hh);
            if($this->activation){
                $calcState->ac_hh = new \stdClass();
                $x_hh = $this->activation->forward($calcState->ac_hh,$x_hh,$training);
            }
        } else { // if(reset_after==false)
            // z = s(prev_h dot hwz + hbz)
            // r = s(prev_h dot hwr + hbr)
            $x_z = $K->gemm($prev_h, $this->r_kernel_z,1.0,1.0,$x_z);
            $x_r = $K->gemm($prev_h, $this->r_kernel_r,1.0,1.0,$x_r);
            if($this->recurrentActivation){
                $calcState->ac_z = new \stdClass();
                $x_z = $this->recurrentActivation->forward($calcState->ac_z,$x_z,$training);
                $calcState->ac_r = new \stdClass();
                $x_r = $this->recurrentActivation->forward($calcState->ac_r,$x_r,$training);
            }
            // hh = t((r * prev_h) dot hwhh + hh)
            $x_r_prev_r = $K->mul($x_r,$prev_h);
            $x_hh = $K->gemm($x_r_prev_r, $this->r_kernel_hh,1.0,1.0,$x_hh);
            if($this->activation){
                $calcState->ac_hh = new \stdClass();
                $x_hh = $this->activation->forward($calcState->ac_hh,$x_hh,$training);
            }
        }
        // next_h = z * prev_h +  (1-z) * hh
        $x1_z = $K->increment($K->scale(-1,$x_z),1);
        $next_h = $K->add(
            $K->mul($x_z,$prev_h),
            $K->mul($x1_z,$x_hh));

        $calcState->inputs = $inputs;
        $calcState->prev_h = $prev_h;
        $calcState->x_z = $x_z;
        $calcState->x1_z = $x1_z;
        $calcState->x_r = $x_r;
        $calcState->x_hh = $x_hh;
        if($this->resetAfter) {
            $calcState->internal_hh = $internal_hh;
        } else {
            $calcState->x_r_prev_r = $x_r_prev_r;
        }
        return [$next_h,[$next_h]];
    }

    protected function differentiate(NDArray $dOutputs, array $dStates, object $calcState) : array
    {
        $K = $this->backend;
        $dNext_h = $dStates[0];
        $dNext_h = $K->add($dOutputs,$dNext_h);

        // forward:
        //  next_h = z * prev_h + (1-z) * hh
        // backward:
        // dprev_h = dnext_h * z
        // d_hh = dnext_h * (1-z)
        // d_z = dnext_h * (prev_h - hh)
        $dPrev_h = $K->mul($dNext_h,$calcState->x_z);
        $dX_hh = $K->mul($dNext_h,$calcState->x1_z);
        $dX_z = $K->mul($dNext_h,$K->sub($calcState->prev_h,$calcState->x_hh));

        if($this->resetAfter) {
            // hh output
            if($this->activation){
                $dX_hh = $this->activation->backward($calcState->ac_hh,$dX_hh);
            }
            // forward:
            // hhx = (inputs dot Wk)+b1
            // internal_hh = (prev_h dot Wh)+b2
            // hh = hhx + r*internal_hh
            //
            // backward:
            // d_internal_hh = d_hh * r
            // d_r = d_hh * internal_hh
            $d_internal_hh = $K->mul($dX_hh,$calcState->x_r);
            $dX_r = $K->mul($dX_hh,$calcState->internal_hh);

            // z gate
            if($this->recurrentActivation){
                $dX_z = $this->recurrentActivation->backward($calcState->ac_z,$dX_z);
            }
            // forward:
            // zx = (inputs dot Wk)+b1
            // internal_x = (prev_h dot Wh)+b2
            // z = zx + internal_x
            // backward:
            // d_zx = d_z
            // d_internal_z = d_z
            // r gate
            if($this->recurrentActivation){
                $dX_r = $this->recurrentActivation->backward($calcState->ac_r,$dX_r);
            }
            // forward:
            // rx = (inputs dot Wk)+b1
            // internal_r = (prev_h dot Wh)+b2
            // r = rx + internal_r
            // backward:
            // d_rx = d_r
            // d_internal_r = d_r

            // recurrent dot
            // forward
            // recurrent_kernel = concat(internal_x,internal_r,internal_hh)
            // internalOutput = h_prev dot
            //     recurrent_kernel
            // backward:
            // d_recurrent_kernel =
            //     h_prev.T dot d_internaloutput
            // dh_prev = d_internaloutput
            //     dot recurrent_kernel.T
            $dInternalOutput = $K->stack(
                [$dX_z,$dX_r,$d_internal_hh],$axis=1);
            $shape = $dInternalOutput->shape();
            $batches = array_shift($shape);
            $dInternalOutput = $dInternalOutput->reshape([
                    $batches,
                    array_product($shape)
            ]);

            if($this->dRecurrentBias) {
                $K->update_add($this->dRecurrentBias,$K->sum($dInternalOutput, $axis=0));
            }
            $K->gemm($calcState->prev_h, $dInternalOutput,1.0,1.0,
                $this->dRecurrentKernel,true,false);
            $K->gemm($dInternalOutput, $this->recurrentKernel,1.0,1.0,
                $dPrev_h,false,true);
        } else {
            // hh output
            if($this->activation){
                $dX_hh = $this->activation->backward($calcState->ac_hh,$dX_hh);
            }
            $K->gemm($calcState->x_r_prev_r, $dX_hh, 1.0,1.0,$this->dR_kernel_hh,true,false);
            $dhh_r = $K->gemm($dX_hh, $this->r_kernel_hh,1.0,0.0,null,false,true);
            $K->update_add($dPrev_h,$K->mul($calcState->x_r,$dhh_r));

            // z gate
            if($this->recurrentActivation){
                $dX_z = $this->recurrentActivation->backward($calcState->ac_z,$dX_z);
            }
            $K->gemm($calcState->prev_h, $dX_z,1.0,1.0,$this->dR_kernel_z,true,false);
            $K->gemm($dX_z, $this->r_kernel_z,1.0,1.0,$dPrev_h,false,true);

            // r gate
            $dX_r = $K->mul($dhh_r,$calcState->prev_h);
            if($this->recurrentActivation){
                $dX_r = $this->recurrentActivation->backward($calcState->ac_r,$dX_r);
            }
            $K->gemm($calcState->prev_h, $dX_r,1.0,1.0,$this->dR_kernel_r,true,false);
            $K->gemm($dX_r, $this->r_kernel_r,1.0,1.0,$dPrev_h,false,true);
        }

        // stack diff gate outputs
        $dGateOuts = $K->stack(
            [$dX_z,$dX_r,$dX_hh],$axis=1);
        $shape = $dGateOuts->shape();
        $batches = array_shift($shape);
        $dGateOuts = $dGateOuts->reshape([
                $batches,
                array_product($shape)
        ]);

        if($this->dInputBias) {
            $K->update_add($this->dInputBias,$K->sum($dGateOuts, $axis=0));
        }
        $K->gemm($calcState->inputs, $dGateOuts,1.0,1.0,$this->dKernel,true,false);
        $dInputs = $K->gemm($dGateOuts, $this->kernel,1.0,1.0,null,false,true);
        return [$dInputs,[$dPrev_h]];
    }
}
