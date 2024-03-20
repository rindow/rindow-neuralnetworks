<?php
namespace Rindow\NeuralNetworks\Gradient\Core;

use InvalidArgumentException;
use LogicException;
use Rindow\NeuralNetworks\Gradient\Scalar as ScalarInterface;
use Interop\Polite\Math\Matrix\NDArray;

class Scalar implements ScalarInterface
{
    protected $value;

    public function __construct(mixed $value)
    {
        if($value instanceof NDArray) {
            if($value->ndim()!=0) {
                throw new InvalidArgumentException('value must be scalar.');
            }
            $value = $value->toArray();
        } elseif(!is_numeric($value)) {
            throw new InvalidArgumentException('value must be scalar.');
        }
        $this->value = $value;
    }

    public function value() : mixed
    {
        return $this->value;
    }
}
