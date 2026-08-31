<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;
use Rindow\NeuralNetworks\Gradient\Core\NullValue;

class Gather extends AbstractFunction
{
    protected int $numOfInputs = 2;

    protected ?int $axis;
    protected ?int $batchDims;
    protected ?int $detailDepth;
    protected ?int $indexDepth;
    //protected int $reduceNumClass;

    public function __construct(
        object $backend,
        ?int $axis=null,
        ?int $batchDims=null,
        ?int $detailDepth=null,
        ?int $indexDepth=null,
        ?string $name=null,
    )
    {
        parent::__construct($backend,name:$name);
        $this->axis = $axis;
        $this->batchDims = $batchDims;
        $this->detailDepth = $detailDepth;
        $this->indexDepth = $indexDepth;
    }

    protected function call(array $inputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        [$params,$indices] = $inputs;
        //$outputs = $K->gather($source,$indices,$this->realAxis);
        $outputs = $K->gatherb(
            $params,
            $indices,
            axis:$this->axis,
            batchDims:$this->batchDims,
            detailDepth:$this->detailDepth,
            indexDepth:$this->indexDepth,
        );
        $container->indices = $indices;
        $container->orignalParamsShape = $params->shape();
        return [$outputs];
    }

    protected function differentiate(array $dOutputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        //$dSource = $K->scatter(
        //    $container->indices,
        //    $dOutputs,
        //    $this->reduceNumClass,
        //    $this->realAxis
        //);
        $dSource = $K->scatterb(
            $container->indices,
            $dOutputs[0],
            $container->orignalParamsShape,
            axis:$this->axis,
            batchDims:$this->batchDims,
            detailDepth:$this->detailDepth,
            indexDepth:$this->indexDepth,
        );
        $dIndex = new NullValue();
        return [$dSource,$dIndex];
    }

}
