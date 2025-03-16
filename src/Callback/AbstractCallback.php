<?php
namespace Rindow\NeuralNetworks\Callback;

use Rindow\NeuralNetworks\Model\Model;

/**
 *
 */
abstract class AbstractCallback implements Callback
{
    protected ?Model $model=null;
    public function __construct(?Model $model=null)
    {
        if($model) {
            $this->setModel($model);
        }
    }

    public function setModel(Model $model) : void
    {
        $this->model = $model;
    }

    public function getModel() : ?Model  
    {
        return $this->model;
    }

    public function onTrainBegin(?array $metrics=null) : void {}
    public function onTestBegin(?array $metrics=null) : void {}
    public function onPredictBegin(?array $metrics=null) : void {}
    public function onTrainEnd(?array $metrics=null) : void {}
    public function onTestEnd(?array $metrics=null) : void {}
    public function onPredictEnd(?array $metrics=null) : void {}
    public function onTrainBatchBegin(int $batch, ?array $metrics=null) : void {}
    public function onTestBatchBegin(int $batch, ?array $metrics=null) : void {}
    public function onTrainBatchEnd(int $batch, ?array $metrics=null) : void {}
    public function onTestBatchEnd(int $batch, ?array $metrics=null) : void {}
    public function onEpochBegin(int $epoch, ?array $metrics=null) : void {}
    public function onEpochEnd(int $epoch, ?array $metrics=null) : void {}
}
