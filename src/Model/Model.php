<?php
namespace Rindow\NeuralNetworks\Model;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Module;
use Rindow\NeuralNetworks\Data\Dataset\Dataset;

interface Model extends Module
{
    public function compile(
        string|object $optimizer=null,
        string|object $loss=null,
        array $metrics=null,
        int $numInputs=null,
    ) : void;

    public function fit(
        $inputs,
        NDArray $tests=null,
        int $batch_size=null,
        int $epochs=null,
        int $verbose=null,
        array|Dataset $validation_data=null,
        array $callbacks=null,
        bool $shuffle=null,
        object $filter=null,
    ) : array;

    public function evaluate(
        $inputs,
        NDArray $trues=null, 
        int $batch_size=null,
        int $verbose=null,
        object|array $callbacks=null,
    ) : array;

    public function predict(
        $inputs, 
        array|object $callbacks=null,
        ...$options
    ) : NDArray;

    public function build(...$inputShapes) : void;
    public function saveWeights(&$modelWeights,$portable=null) : void;
    public function loadWeights($modelWeights) : void;
}
