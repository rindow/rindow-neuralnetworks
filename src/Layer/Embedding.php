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
    protected mixed $embeddingsInitializer;
    protected ?string $embeddingsInitializerName;
    protected ?int $inputDtype=NDArray::int32;
    protected bool $maskZero;

    protected ?NDArray $kernel=null;
    protected NDArray $dKernel;
    //protected $inputs;
    //protected $originalShape;
    //protected $flattenOutputsShape;

    public function __construct(
        object $backend,
        int $inputDim,
        int $outputDim,
        ?int $input_length=null,
        string|callable|null $embeddings_initializer=null,
        ?bool $mask_zero=null,
        ?string $name=null,
    )
    {
        // defaults
        $embeddings_initializer ??= 'random_uniform';
        $mask_zero ??= false;
        
        parent::__construct($backend);
        $K = $backend;
        if($input_length!=null){
            $this->inputShape = [$input_length];
        }
        $this->inputLength = $input_length;
        $this->inputDim = $inputDim;
        $this->outputDim = $outputDim;
        $this->embeddingsInitializer = $K->getInitializer($embeddings_initializer);
        $this->embeddingsInitializerName = $this->toStringName($embeddings_initializer);
        $this->maskZero = $mask_zero;
        $this->initName($name,'embedding');
        $this->allocateWeights(['kernel']);
    }

    public function build(mixed $variable=null, ?array $sampleWeights=null) : void
    {
        $K = $this->backend;
        $embeddingsInitializer = $this->embeddingsInitializer;

        $inputShape = $this->normalizeInputShape($variable);
        if(count($inputShape)!=1) {
            throw new InvalidArgumentException(
                'Unsuppored input shape: ['.implode(',',$inputShape).']');
        }
        if($this->kernel===null) {
            if($sampleWeights) {
                $this->kernel = $sampleWeights[0];
            } else {
                $this->kernel = $embeddingsInitializer(
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
                'embeddings_initializer' => $this->embeddingsInitializerName,
            ]
        ];
    }


    protected function call(NDArray $inputs, ?bool $training=null) : NDArray
    {
        // inputs:  [batch,len]
        // kernel:  [inputDim,outputDim] (numClass=inputDim)
        // outputs: [batch,len,outputDim]

        $K = $this->backend;
        $container = $this->container();
        $container->originalShape = $inputs->shape();
        $inputs = $inputs->reshape([$inputs->size()]);
        $container->reshapedInputs = $inputs;

        // gatherbAdd version
        // kernel:  [inputDim,outputDim]          (numClass=inputDim, len=outputDim)
        // inputs:  [batch,len]                   (n=batch*len)
        // outputs: [batch,len,outputDim]         (n=batch*len, len=outputDim)
        $outputs = $K->gatherb(
            $this->kernel,                  // params
            $inputs,                        // indices
            axis:0,
            batchDims:0,
            detailDepth:1,
            indexDepth:1,
        );
        $container->reshapedOutputsShape = $outputs->shape();
        $shape = $container->originalShape;
        array_push($shape,$this->kernel->shape()[1]);
        $outputs = $outputs->reshape($shape);

        //// gatherND version
        //$inputs = $inputs->reshape([$inputs->size(),1]);
        //// gatherND(
        ////  params:  [p0=inputDim,  k=outputDim]   <= kernel
        ////  indices: [n=batch*len,  indexDepth=1]  <= inputs
        ////  outputs: [n=batch*len,  k=outputDim]   <= outputs
        ////  batchDims: 0
        ////)
        //$outputs = $K->gatherND($this->kernel,$inputs);
        //$container->flattenOutputsShape = $outputs->shape();
        //$shape = $container->originalShape;
        //array_push($shape,$this->outputDim);
        //$outputs =$outputs->reshape($shape);

        return $outputs;
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        // dOutputs: [batch,len,outputDim]          (m=batch*len, k=outputDim)
        // inputs:   [batch,len]                    (m=batch*len)
        // kernel:   [inputDim,outputDim]
   
        $K = $this->backend;
        $container = $this->container();

        // scatterbAdd version
        // inputs:  [batch,len]                   (n=batch*len)
        // dOutputs: [batch,len,outputDim]        (n=batch*len, len=outputDim)
        // kernel:  [inputDim,outputDim]          (numClass=inputDim, len=outputDim)
        // === Scatter and ReduceSum edition ===
        // tmp[m,x[m,n=1],k] = dOutputs[m,n=1,k];
        // dKernel[numClass,k] = reduceSum(tmp[m,numClass,k],axis=0);
        // [batch,len]
        $dOutputs = $dOutputs->reshape($container->reshapedOutputsShape);
        $K->clear($this->dKernel);
        $K->scatterbAdd(
            $container->reshapedInputs, // indices
            $dOutputs,                  // updates
            $this->dKernel->shape(),    // shape
            axis:0,
            batchDims:0,
            detailDepth:1,
            indexDepth:1,
            outputs:$this->dKernel
        );
        
        //// scatterNDAdd version
        //// dOutputs: [batch,len,outputDim]          (m=batch*len, k=outputDim)
        //// inputs:   [batch,len]                    (m=batch*len)
        //// scatter:  [batch,len,inputDim,outputDim] (m=batch*len, numClass=inputDim, k=outputDim)
        //// kernel:   [inputDim,outputDim]
        //// === Scatter and ReduceSum edition ===
        //// tmp[m,x[m,n=1],k] = dOutputs[m,n=1,k];
        //// dKernel[numClass,k] = reduceSum(tmp[m,numClass,k],axis=0);
        //// [batch,len]
        //$n = (int)array_product($container->originalShape);
        //$outputDim = $this->kernel->shape()[1];
        //$inputs = $container->inputs->reshape([$n,1]);
        //$dOutputs = $dOutputs->reshape([$n,$outputDim]);
        //$shape = $this->kernel->shape();
        //// scatterNDAdd(
        ////  indices: [m=1, n=batch*len, 1]
        ////  updates: [m=1, n=batch*len, k=outputDim]
        ////  outputs: [m=1, p0=inputDim, k=outputDim]
        ////  batchDims: 0
        //// )
        //$K->clear($this->dKernel);
        //$K->scatterNDAdd($inputs, $dOutputs, shape:$shape, batchDims:0,outputs:$this->dKernel);

        return $K->zeros(
            $container->originalShape,
            dtype:$container->reshapedInputs->dtype()
        );//dummy
    }

    public function computeMask(
        array|NDArray $inputs,
        array|NDArray|null $previousMask
        ) : array|NDArray|null
    {
        $K = $this->backend;
        if(!$this->maskZero) {
            return $previousMask;
        }
        if(!($inputs instanceof NDArray)) {
            throw new InvalidArgumentException('inputs must be NDArray');
        }
        $mask = $K->cast($inputs,NDArray::bool);
        return $mask;
    }

    //protected function call(NDArray $inputs, ?bool $training=null) : NDArray
    //{
    //    $K = $this->backend;
    //    $container = $this->container();
    //    $container->originalShape = $inputs->shape();
    //    $container->inputs = $inputs;
    //    $inputs = $inputs->reshape([$inputs->size()]);
    //    $outputs = $K->gather($this->kernel,$inputs);
    //    $container->flattenOutputsShape = $outputs->shape();
    //    $shape = $container->originalShape;
    //    array_push($shape,$this->outputDim);
    //    return $outputs->reshape($shape);
    //}
//
    //protected function differentiate(NDArray $dOutputs) : NDArray
    //{
    //    $K = $this->backend;
    //    $container = $this->container();
    //    $inputs = $container->inputs;
    //    $dOutputs = $dOutputs->reshape($container->flattenOutputsShape);
    //    $inputs = $inputs->reshape([$inputs->size()]);
    //    $K->clear($this->dKernel);
    //    $K->scatterAdd($this->dKernel,$inputs, $dOutputs);
    //    return $container->inputs->reshape($container->originalShape);//dummy
    //}

}
