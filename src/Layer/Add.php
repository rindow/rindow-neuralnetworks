<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class Add extends AbstractMultiInputLayer
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
        $this->initName($name,'add');
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
        [$shapeX,$shapeY] = $inputShapes;
        if($shapeX!=$shapeY) {
            throw new InvalidArgumentException('Inputs have incompatible shapes. Received shapes '.$this->shapeToString($shapeX).' and '.$this->shapeToString($shapeY));
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
        if(count($inputs)!=2) {
            throw new InvalidArgumentException('num of inputs must be 2 values: '.count($inputs).' value gives.');
        }
        [$x,$y] = $inputs;
        if($x->shape()!=$y->shape()) {
            throw new InvalidArgumentException('Inputs have incompatible shapes. Received shapes '.$this->shapeToString($x->shape()).' and '.$this->shapeToString($y->shape()));
        }
        return $K->add($x,$y);
    }

    protected function differentiate(NDArray $dOutputs) : array
    {
        $K = $this->backend;
        return [$dOutputs,$dOutputs];
    }

    public function computeMask(
        array|NDArray $inputs,
        array|NDArray|null $previousMask
        ) : array|NDArray|null
    {
        $K = $this->backend;
        if($previousMask==null) {
            return null;
        }
        if(!is_array($previousMask)) {
            throw new InvalidArgumentException('num of masks must be 2 items: 1 mask gives.');
        }
        if(count($previousMask)!=2) {
            throw new InvalidArgumentException('num of masks must be 2 items: '.count($previousMask).' mask gives.');
        }
        [$maskX,$maskY] = $previousMask;
        if($maskX===null) {
            return $maskY;
        }
        if($maskY===null) {
            return $maskX;
        }
        $mask = $K->masking($maskX,$maskY);
        return [$mask];
    }

}
