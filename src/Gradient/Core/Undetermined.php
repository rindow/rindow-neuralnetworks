<?php
namespace Rindow\NeuralNetworks\Gradient\Core;

use InvalidArgumentException;
use LogicException;
use Interop\Polite\Math\Matrix\NDArray;

class Undetermined extends Variable
{
    public function __construct($value=null)
    {
        $this->setValue($value);
    }

    public function setValue(UndeterminedNDArray $value=null)
    {
        $this->value = $value;
    }
}
