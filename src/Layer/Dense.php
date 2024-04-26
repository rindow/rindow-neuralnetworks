<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class Dense extends AbstractLayer
{
    use GenericUtils;
    protected int $units;
    protected bool $useBias;
    protected mixed $kernelInitializer;
    protected mixed $biasInitializer;
    protected ?string $kernelInitializerName;
    protected ?string $biasInitializerName;

    protected ?NDArray $kernel=null;
    protected NDArray $bias;
    protected NDArray $dKernel;
    protected NDArray $dBias;
    //protected $inputs;

    /**
     * @param array<int> $input_shape
     */
    public function __construct(
        object $backend,
        int $units,
        array $input_shape=null,
        string|object $activation=null,
        bool $use_bias=null,
        string|callable $kernel_initializer=null,
        string|callable $bias_initializer=null,
        string $name=null,
    )
    {
        // defaults
        $input_shape = $input_shape ?? null;
        $activation = $activation ?? null;
        $use_bias = $use_bias ?? true;
        $kernel_initializer = $kernel_initializer ?? 'glorot_uniform';
        $bias_initializer = $bias_initializer ?? 'zeros';
        $name = $name ?? null;
        //'kernel_regularizer'=>null, 'bias_regularizer'=>null,
        //'activity_regularizer'=null,
        //'kernel_constraint'=null, 'bias_constraint'=null,

        parent::__construct($backend);
        $K = $backend;
        $this->units = $units;
        $this->inputShape = $input_shape;
        $this->kernelInitializer = $K->getInitializer($kernel_initializer);
        $this->biasInitializer   = $K->getInitializer($bias_initializer);
        $this->kernelInitializerName = $this->toStringName($kernel_initializer);
        $this->biasInitializerName = $this->toStringName($bias_initializer);
        $this->useBias = $use_bias;
        $this->initName($name,'dense');
        $this->allocateWeights($this->useBias?2:1);
        $this->setActivation($activation);
    }

    public function build(mixed $variable=null, array $sampleWeights=null) : void
    {
        $K = $this->backend;
        $kernelInitializer = $this->kernelInitializer;
        $biasInitializer = $this->biasInitializer;

        $inputShape = $this->normalizeInputShape($variable);
        //if(count($inputShape)!=1) {
        //    throw new InvalidArgumentException(
        ///        'Unsuppored input shape: ['.implode(',',$inputShape).']');
        //}
        $shape = $inputShape;
        $inputDim=array_pop($shape);
        if($this->kernel===null) {
            if($sampleWeights) {
                $this->kernel = $sampleWeights[0];
                $this->bias = $sampleWeights[1];
            } else {
                $this->kernel = $kernelInitializer(
                    [$inputDim,$this->units],
                    [$inputDim,$this->units]);
                if($this->useBias) {
                    $this->bias = $biasInitializer([$this->units]);
                }
            }
        }

        $this->dKernel = $K->zerosLike($this->kernel);
        if($this->useBias) {
            $this->dBias = $K->zerosLike($this->bias);
        }
        array_push($shape,$this->units);
        $this->outputShape = $shape;
        $this->syncWeightVariables();
    }

    public function getParams() : array
    {
        if($this->useBias) {
            return [$this->kernel,$this->bias];
        } else {
            return [$this->kernel];
        }
    }

    public function getGrads() : array
    {
        if($this->useBias) {
            return [$this->dKernel,$this->dBias];
        } else {
            return [$this->dKernel];
        }
    }

    public function reverseSyncWeightVariables() : void
    {
        if($this->useBias) {
            $this->kernel = $this->weights[0]->value();
            $this->bias = $this->weights[1]->value();
        } else {
            $this->kernel = $this->weights[0]->value();
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

    protected function call(NDArray $inputs, bool $training=null) : NDArray
    {
        $K = $this->backend;
        $container = $this->container();
        $shape = $inputs->shape();
        $container->origInputsShape = $shape;
        $inputDim=array_pop($shape);
        $inputSize=array_product($shape);
        $container->inputs = $inputs->reshape([$inputSize,$inputDim]);
        if($this->useBias) {
            $outputs = $K->batch_gemm($container->inputs, $this->kernel,1.0,1.0,$this->bias);
        } else {
            $outputs = $K->gemm($container->inputs, $this->kernel);
        }
        $container->flattenOutputsShape = $outputs->shape();
        array_push($shape,$this->units);
        $outputs = $outputs->reshape($shape);
        if($this->activation) {
            $container->activation = new \stdClass();
            $outputs = $this->activation->forward($container->activation,$outputs,training:$training);
        }
        return $outputs;
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        $container = $this->container();
        if($this->activation) {
            $dOutputs = $this->activation->backward($container->activation,$dOutputs);
        }
        $dInputs = $K->zerosLike($container->inputs);
        $dOutputs=$dOutputs->reshape($container->flattenOutputsShape);
        $K->gemm($dOutputs, $this->kernel,1.0,0.0,$dInputs,false,true);

        // update params
        $K->gemm($container->inputs, $dOutputs,1.0,0.0,$this->dKernel,true,false);
        if($this->useBias)
            $K->sum($dOutputs, axis:0,output:$this->dBias);

        return $dInputs->reshape($container->origInputsShape);
    }

}
