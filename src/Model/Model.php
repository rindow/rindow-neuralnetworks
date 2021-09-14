<?php
namespace Rindow\NeuralNetworks\Model;

use Interop\Polite\Math\Matrix\NDArray;

interface Model
{
    public function compile(array $options=null) : void;
    public function fit(NDArray $inputs, NDArray $tests, array $options=null) : array;
    public function evaluate(NDArray $x, NDArray $t, array $options=null) : array;
    public function predict(NDArray $inputs, array $options=null) : NDArray;
    public function saveWeights(&$modelWeights,$portable=null) : void;
    public function loadWeights($modelWeights) : void;
}
