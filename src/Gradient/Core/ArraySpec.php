<?php
namespace Rindow\NeuralNetworks\Gradient\Core;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\ArraySpec as ArraySpecInterface;
use Rindow\NeuralNetworks\Gradient\ArrayShape as ArrayShapeInterface;

class ArraySpec implements ArraySpecInterface
{
    protected ArrayShapeInterface $shape;
    protected int $dtype=NDArray::float32;
    protected ?string $name=null;

    /**
     * @param array<int>|ArrayShapeInterface $shape
     */
    public function __construct(
        ArrayShapeInterface|array $shape,
        ?int $dtype=null,
        ?string $name=null,
    ) {
        if(is_array($shape)) {
            $shape = new ArrayShape($shape);
        }
        $dtype ??= NDArray::float32;
        $this->shape = $shape;
        $this->dtype = $dtype;
        $this->name = $name;
    }

    public function shape() : ArrayShapeInterface
    {
        return $this->shape;
    }

    public function dtype() : int
    {
        return $this->dtype;
    }

    public function name() : ?string
    {
        return $this->name;
    }
}
