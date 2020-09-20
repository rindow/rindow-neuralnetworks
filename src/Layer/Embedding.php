<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class Embedding extends AbstractLayer implements Layer
{
    use GenericUtils;
    protected $backend;
    protected $inputDim;
    protected $outputDim;
    protected $kernelInitializer;
    protected $input_length;

    protected $kernel;
    protected $dKernel;
    protected $inputs;
    protected $originalShape;
    protected $flattenOutputsShape;

    public function __construct($backend,int $inputDim,int $outputDim, array $options=null)
    {
        extract($this->extractArgs([
            'input_length'=>null,
            'kernel_initializer'=>'random_uniform',
        ],$options));
        $this->backend = $K = $backend;
        if($input_length!=null){
            $this->inputShape = [$input_length];
        }
        $this->inputLength = $input_length;
        $this->inputDim = $inputDim;
        $this->outputDim = $outputDim;
        $this->kernelInitializer = $K->getInitializer($kernel_initializer);
        $this->kernelInitializerName = $kernel_initializer;
    }

    public function build(array $inputShape=null, array $options=null) : array
    {
        extract($this->extractArgs([
            'sampleWeights'=>null,
        ],$options));
        $K = $this->backend;
        $kernelInitializer = $this->kernelInitializer;

        $inputShape = $this->normalizeInputShape($inputShape);
        if(count($inputShape)!=1) {
            throw new InvalidArgumentException(
                'Unsuppored input shape: ['.implode(',',$inputShape).']');
        }
        if($sampleWeights) {
            $this->kernel = $sampleWeights[0];
        } else {
            $this->kernel = $kernelInitializer(
                [$this->inputDim,$this->outputDim],
                [$this->inputDim,$this->outputDim]
            );
        }
        $this->dKernel = $K->zerosLike($this->kernel);
        $this->outputShape = array_merge($inputShape,[$this->outputDim]);
        return $this->outputShape;
    }

    public function getParams() : array
    {
        return [$this->kernel];
    }

    public function getGrads() : array
    {
        return [$this->dKernel];
    }

    public function getConfig() : array
    {
        return [
            'inputDim' => $this->inputDim,
            'outputDim' => $this->outputDim,
            'options' => [
                'input_length'=>$this->input_length,
                'kernel_initializer' => $this->kernelInitializerName,
            ]
        ];
    }

    protected function call(NDArray $inputs, bool $training) : NDArray
    {
        $K = $this->backend;
        $this->originalShape = $inputs->shape();
        $this->inputs = $inputs->reshape(
            [$inputs->size()]);
        $outputs = $K->select($this->kernel,$this->inputs,$axis=0);
        $this->flattenOutputsShape = $outputs->shape();
        $shape = $this->originalShape;
        array_push($shape,$this->outputDim);
        return $outputs->reshape($shape);
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        $dOutputs = $dOutputs->reshape($this->flattenOutputsShape);
        $K->clear($this->dKernel);
        $K->scatterAdd($this->dKernel,$this->inputs, $dOutputs);
        return $this->inputs->reshape($this->originalShape);//dummy
    }
}
