<?php
namespace Rindow\NeuralNetworks\Layer;

use Interop\Polite\Math\Matrix\NDArray;

/**
 *
 */
interface RNNLayer extends LayerBase
{
    public function forward(NDArray $inputs, bool $training,array $initialStates=null,array $options=null) ;
    public function backward(NDArray $dOutputs, array $dStates=null) ;
}