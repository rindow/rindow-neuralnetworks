<?php
namespace Rindow\NeuralNetworks\Model;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Gradient\Module;
use Rindow\NeuralNetworks\Data\Dataset\Dataset;
use Rindow\NeuralNetworks\Callback\Callback;
use Rindow\NeuralNetworks\Callback\Broadcaster;
use Rindow\NeuralNetworks\Metric\Metric;
use Rindow\NeuralNetworks\Layer\Layer;

interface Model extends Module
{
    /**
     * @return array<Layer>
     */
    public function layers() : array;

    /**
     * @return array<Variable>
     */
    public function parameterVariables() : array;

    /**
     * @param array<int|string,string|Metric> $metrics
     */
    public function compile(
        string|object $optimizer=null,
        string|object $loss=null,
        array $metrics=null,
        int $numInputs=null,
    ) : void;

    /**
     * @param array<Callback> $callbacks
     * @param array{mixed,mixed}|Dataset<NDArray> $validation_data
     * @return array<string,array<float>>
     */
    public function fit(
        mixed $inputs,
        NDArray $tests=null,
        int $batch_size=null,
        int $epochs=null,
        int $verbose=null,
        array|Dataset $validation_data=null,
        array $callbacks=null,
        bool $shuffle=null,
        object $filter=null,
    ) : array;

    /**
     * @param array<Callback>|Broadcaster $callbacks
     * @return array<string,float>
     */
    public function evaluate(
        mixed $inputs,
        NDArray $trues=null, 
        int $batch_size=null,
        int $verbose=null,
        array|Broadcaster $callbacks=null,
    ) : array;

    /**
     * @param array<Callback>|Broadcaster $callbacks
     * @param mixed ...$options
     */
    public function predict(
        mixed $inputs, 
        array|Broadcaster $callbacks=null,
        mixed ...$options
    ) : NDArray;

    /**
     * @param array<int>|NDArray|Variable ...$inputShapes
     */
    public function build(array|NDArray|Variable ...$inputShapes) : void;

    /**
     * @param array<mixed,mixed> $modelWeights
     */
    public function saveWeights(iterable &$modelWeights,bool $portable=null) : void;

    /**
     * @param array<mixed,mixed> $modelWeights
     */
    public function loadWeights(iterable $modelWeights) : void;
}
