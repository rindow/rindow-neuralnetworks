<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class InheritMask extends AbstractMultiInputLayer
{
    use GenericUtils;

    /**
     * @param array<array<int>> $input_shapes
     */
    public function __construct(
        object $backend,
        ?array $input_shapes=null,
        ?string $name=null,
    )
    {
        parent::__construct($backend);
        $this->inputShape = $input_shapes;
        $this->initName($name,'inheritmask');
    }

    public function build(mixed $variables=null, ?array $sampleWeights=null) : void
    {
        $K = $this->backend;
        if(!is_array($variables) && $variables!==null) {
            throw new InvalidArgumentException('inputs must be list of variable');
        }
        $inputShapes = $this->normalizeInputShapes($variables);
        if(count($inputShapes)!=2) {
            throw new InvalidArgumentException('num of inputs must be 2 values: input dims is '.count($inputShapes));
        }
        $this->outputShape = $inputShapes[0];
    }

    public function getParams() : array
    {
        return [];
    }

    public function getGrads() : array
    {
        return [];
    }

    public function getConfig() : array
    {
        return [
        ];
    }

    protected function call(array $inputs, ?bool $training=null) : NDArray
    {
        $K = $this->backend;
        $container = $this->container();
        $container->maskSourceShape = $inputs[1]->shape();
        $container->maskSourceDtype = $inputs[1]->dtype();
        return $inputs[0];
    }

    protected function differentiate(NDArray $dOutputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        $dInputs = [
            $dOutputs,
            $K->zeros($container->maskSourceShape,$container->maskSourceDtype),
        ];
        return $dInputs;
    }

    public function computeMask(
        array|NDArray $inputs,
        array|NDArray|null $previousMask
        ) : array|NDArray|null
    {
        if($previousMask==null) {
            return null;
        }
        return [$previousMask[1]];
    }

}
