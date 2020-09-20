<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class Dense extends AbstractLayer implements Layer
{
    use GenericUtils;
    protected $backend;
    protected $units;
    protected $useBias;
    protected $kernelInitializer;
    protected $biasInitializer;
    protected $kernelInitializerName;
    protected $biasInitializerName;

    protected $kernel;
    protected $bias;
    protected $dKernel;
    protected $dBias;
    protected $inputs;

    public function __construct($backend,int $units, array $options=null)
    {
        extract($this->extractArgs([
            'input_shape'=>null,
            'activation'=>null,
            'use_bias'=>true,
            'kernel_initializer'=>'glorot_uniform',
            'bias_initializer'=>'zeros',
            //'kernel_regularizer'=>null, 'bias_regularizer'=>null,
            //'activity_regularizer'=null,
            //'kernel_constraint'=null, 'bias_constraint'=null,
        ],$options));
        $this->backend = $K = $backend;
        $this->units = $units;
        $this->inputShape = $input_shape;
        $this->kernelInitializer = $K->getInitializer($kernel_initializer);
        $this->biasInitializer   = $K->getInitializer($bias_initializer);
        $this->kernelInitializerName = $kernel_initializer;
        $this->biasInitializerName = $bias_initializer;
        if($use_bias===null || $use_bias) {
            $this->useBias = true;
        }
        $this->setActivation($activation);
    }

    public function build(array $inputShape=null, array $options=null) : array
    {
        extract($this->extractArgs([
            'sampleWeights'=>null,
        ],$options));
        $K = $this->backend;
        $kernelInitializer = $this->kernelInitializer;
        $biasInitializer = $this->biasInitializer;

        $inputShape = $this->normalizeInputShape($inputShape);
        //if(count($inputShape)!=1) {
        //    throw new InvalidArgumentException(
        ///        'Unsuppored input shape: ['.implode(',',$inputShape).']');
        //}
        $shape = $inputShape;
        $this->inputDim=array_pop($shape);
        if($sampleWeights) {
            $this->kernel = $sampleWeights[0];
            $this->bias = $sampleWeights[1];
        } else {
            $this->kernel = $kernelInitializer(
                [$this->inputDim,$this->units],
                [$this->inputDim,$this->units]);
            if($this->useBias) {
                $this->bias = $biasInitializer([$this->units]);
            }
        }
        $this->dKernel = $K->zerosLike($this->kernel);
        if($this->useBias) {
            $this->dBias = $K->zerosLike($this->bias);
        }
        array_push($shape,$this->units);
        $this->outputShape = $shape;
        return $this->outputShape;
    }

    public function getParams() : array
    {
        if($this->bias) {
            return [$this->kernel,$this->bias];
        } else {
            return [$this->kernel];
        }
    }

    public function getGrads() : array
    {
        if($this->bias) {
            return [$this->dKernel,$this->dBias];
        } else {
            return [$this->dKernel];
        }
    }

    public function getConfig() : array
    {
        return [
            'units' => $this->units,
            'options' => [
                'input_shape'=>$this->inputShape,
                'use_bias'=>$this->useBias,
                'kernel_initializer' => $this->kernelInitializerName,
                'bias_initializer' => $this->biasInitializerName,
                'activation'=>$this->activationName,
            ]
        ];
    }

    protected function call(NDArray $inputs, bool $training) : NDArray
    {
        $K = $this->backend;
        $shape = $inputs->shape();
        $this->origInputsShape = $shape;
        $inputDim=array_pop($shape);
        $inputSize=array_product($shape);
        $this->inputs = $inputs->reshape([$inputSize,$inputDim]);
        if($this->bias) {
            $outputs = $K->batch_gemm($this->inputs, $this->kernel,1.0,1.0,$this->bias);
        } else {
            $outputs = $K->gemm($this->inputs, $this->kernel);
        }
        $this->flattenOutputsShape = $outputs->shape();
        array_push($shape,$this->units);
        $outputs = $outputs->reshape($shape);
        if($this->activation)
            $outputs = $this->activation->forward($outputs,$training);
        return $outputs;
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        if($this->activation)
            $dOutputs = $this->activation->backward($dOutputs);
        $dInputs = $K->zerosLike($this->inputs);
        $dOutputs=$dOutputs->reshape($this->flattenOutputsShape);
        $K->gemm($dOutputs, $this->kernel,1.0,0.0,$dInputs,false,true);

        // update params
        $K->gemm($this->inputs, $dOutputs,1.0,0.0,$this->dKernel,true,false);
        if($this->dBias)
            $K->copy($K->sum($dOutputs, $axis=0),$this->dBias);

        return $dInputs->reshape($this->origInputsShape);
    }
}
