<?php
namespace Rindow\NeuralNetworks\Callback;

use Rindow\NeuralNetworks\Model\Model;

/**
 *
 */
interface Callback
{
    public function setModel(Model $model);
    public function getModel();
    public function onTrainBegin(array $metrics=null) : void;
    public function onTestBegin(array $metrics=null) : void;
    public function onPredictBegin(array $metrics=null) : void;
    public function onTrainEnd(array $metrics=null) : void;
    public function onTestEnd(array $metrics=null) : void;
    public function onPredictEnd(array $metrics=null) : void;
    public function onTrainBatchBegin(int $batch, array $metrics=null) : void;
    public function onTestBatchBegin(int $batch, array $metrics=null) : void;
    public function onTrainBatchEnd(int $batch, array $metrics=null) : void;
    public function onTestBatchEnd(int $batch, array $metrics=null) : void;
    public function onEpochBegin(int $epoch, array $metrics=null) : void;
    public function onEpochEnd(int $epoch, array $metrics=null) : void;
}
