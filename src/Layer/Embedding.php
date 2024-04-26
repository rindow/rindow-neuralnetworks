<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class Embedding extends AbstractLayer
{
    use GenericUtils;
    protected ?int $inputLength;
    protected int $inputDim;
    protected int $outputDim;
    protected mixed $kernelInitializer;
    protected ?string $kernelInitializerName;

    protected ?NDArray $kernel=null;
    protected NDArray $dKernel;
    //protected $inputs;
    //protected $originalShape;
    //protected $flattenOutputsShape;

    public function __construct(
        object $backend,
        int $inputDim,
        int $outputDim,
        int $input_length=null,
        string|callable $kernel_initializer=null,
        string $name=null,
    )
    {
        // defaults
        $input_length = $input_length ?? null;
        $kernel_initializer = $kernel_initializer ?? 'random_uniform';
        $name = $name ?? null;
        
        parent::__construct($backend);
        $K = $backend;
        if($input_length!=null){
            $this->inputShape = [$input_length];
        }
        $this->inputLength = $input_length;
        $this->inputDim = $inputDim;
        $this->outputDim = $outputDim;
        $this->kernelInitializer = $K->getInitializer($kernel_initializer);
        $this->kernelInitializerName = $this->toStringName($kernel_initializer);
        $this->initName($name,'embedding');
        $this->allocateWeights(1);
    }

    public function build(mixed $variable=null, array $sampleWeights=null) : void
    {
        $K = $this->backend;
        $kernelInitializer = $this->kernelInitializer;

        $inputShape = $this->normalizeInputShape($variable);
        if(count($inputShape)!=1) {
            throw new InvalidArgumentException(
                'Unsuppored input shape: ['.implode(',',$inputShape).']');
        }
        if($this->kernel===null) {
            if($sampleWeights) {
                $this->kernel = $sampleWeights[0];
            } else {
                $this->kernel = $kernelInitializer(
                    [$this->inputDim,$this->outputDim],
                    [$this->inputDim,$this->outputDim]
                );
            }
        }
        $this->dKernel = $K->zerosLike($this->kernel);
        $this->outputShape = array_merge($inputShape,[$this->outputDim]);
        $this->syncWeightVariables();
    }

    public function getParams() : array
    {
        return [$this->kernel];
    }

    public function getGrads() : array
    {
        return [$this->dKernel];
    }

    public function reverseSyncWeightVariables() : void
    {
        $this->kernel = $this->weights[0]->value();
    }

    public function getConfig() : array
    {
        return [
            'inputDim' => $this->inputDim,
            'outputDim' => $this->outputDim,
            'options' => [
                'input_length'=>$this->inputLength,
                'kernel_initializer' => $this->kernelInitializerName,
            ]
        ];
    }

    protected function call(NDArray $inputs, bool $training=null) : NDArray
    {
        $K = $this->backend;
        $container = $this->container();
        $container->originalShape = $inputs->shape();
        $container->inputs = $inputs->reshape(
            [$inputs->size()]);
        $outputs = $K->gather($this->kernel,$container->inputs);
        $container->flattenOutputsShape = $outputs->shape();
        $shape = $container->originalShape;
        array_push($shape,$this->outputDim);
        return $outputs->reshape($shape);
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        $container = $this->container();
        $dOutputs = $dOutputs->reshape($container->flattenOutputsShape);
        $K->clear($this->dKernel);
        $K->scatterAdd($this->dKernel,$container->inputs, $dOutputs);
        return $container->inputs->reshape($container->originalShape);//dummy
    }
}
