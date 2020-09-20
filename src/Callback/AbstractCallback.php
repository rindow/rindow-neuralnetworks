<?php
namespace Rindow\NeuralNetworks\Callback;

use Rindow\NeuralNetworks\Model\Model;

/**
 *
 */
abstract class AbstractCallback implements Callback
{
    public function __construct($model=null)
    {
        if($model)
            $this->setModel($model);
    }

    public function setModel(Model $model)
    {
        $this->model = $model;
    }

    public function getModel()
    {
        return $this->model;
    }

    public function onTrainBegin(array $logs=null) : void {}
    public function onTestBegin(array $logs=null) : void {}
    public function onPredictBegin(array $logs=null) : void {}
    public function onTrainEnd(array $logs=null) : void {}
    public function onTestEnd(array $logs=null) : void {}
    public function onPredictEnd(array $logs=null) : void {}
    public function onTrainBatchBegin(int $batch, array $logs=null) : void {}
    public function onTestBatchBegin(int $batch, array $logs=null) : void {}
    public function onTrainBatchEnd(int $batch, array $logs=null) : void {}
    public function onTestBatchEnd(int $batch, array $logs=null) : void {}
    public function onEpochBegin(int $epoch, array $logs=null) : void {}
    public function onEpochEnd(int $epoch, array $logs=null) : void {}
}
