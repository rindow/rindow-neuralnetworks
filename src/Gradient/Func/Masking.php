<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;
use Rindow\NeuralNetworks\Gradient\Core\NullValue;
use InvalidArgumentException;

class Masking extends AbstractFunction
{
    protected int $numOfInputs = 2;

    protected ?int $batchDims;
    protected ?int $axis;
    protected ?float $fill;
    protected ?int $mode;
    
    public function __construct(
        object $backend,
        ?int $batchDims=null,
        ?int $axis=null,
        ?float $fill=null,
        ?int $mode=null,
        ?string $name=null,
    )
    {
        parent::__construct($backend,name:$name);
        $this->batchDims = $batchDims;
        $this->axis = $axis;
        $this->fill = $fill;
        $this->mode = $mode;
    }

    /**
    *  @param array<NDArray>  $inputs
    *  @return array<NDArray>
    */
    protected function call(array $inputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        [$mask,$data] = $inputs;
        if($mask->dtype()!=NDArray::bool) {
            throw new InvalidArgumentException("mask must be bool type.");
        }
        $container->mask = $mask;

        $output = $K->masking(
            $mask,
            $data,
            batchDims:$this->batchDims,
            axis:$this->axis,
            fill:$this->fill,
            mode:$this->mode,
        );
        return [$output];
    }

    /**
    *  @param array<NDArray>  $dOutputs
    *  @return array<NDArray|NullValue>
    */
    protected function differentiate(array $dOutputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        $mask = $container->mask;

        $dData = $K->masking(
            $mask,
            $dOutputs[0],
            batchDims:$this->batchDims,
            axis:$this->axis,
            fill:0,
            mode:$this->mode,
        );
        $dMask = new NullValue();
        return [$dMask, $dData];
    }
}
