<?php
namespace Rindow\NeuralNetworks\Gradient;

use Traversable;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\Drivers\Service;

interface MaskedNDArray extends NDArray
{
    public function value() : NDArray;
    public function mask() : NDArray;
    public function count() : int;
    /**
     * @return Traversable<NDArray>
     */
    public function getIterator() :  Traversable;
    public function service() : Service;
}
