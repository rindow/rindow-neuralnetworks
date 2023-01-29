<?php
namespace Rindow\NeuralNetworks\Callback;

use Rindow\NeuralNetworks\Model\Model;

/**
 *
 */
class CallbackList
{
    protected $callbacks;
    protected $model;
    public function __construct(Model $model, array $callbacks=null)
    {
        $this->model = $model;
        if($callbacks) {
            foreach ($callbacks as $callback) {
                if(!($callback instanceof Callback))
                    throw new InvalidArgumentException('callbacks must be list of Callback');
                $callback->setModel($model);
            }
            $this->callbacks = $callbacks;
        }
    }

    public function onTrainBegin(array $logs=null) : void
    {
        if($this->callbacks==null)
            return;
        foreach ($this->callbacks as $callback) {
            $callback->onTrainBegin($logs);
        }
    }

    public function onTestBegin(array $logs=null) : void
    {
        if($this->callbacks==null)
            return;
        foreach ($this->callbacks as $callback) {
            $callback->onTestBegin($logs);
        }
    }

    public function onPredictBegin(array $logs=null) : void
    {
        if($this->callbacks==null)
            return;
        foreach ($this->callbacks as $callback) {
            $callback->onPredictBegin($logs);
        }
    }

    public function onTrainEnd(array $logs=null) : void
    {
        if($this->callbacks==null)
            return;
        foreach ($this->callbacks as $callback) {
            $callback->onTrainEnd($logs);
        }
    }

    public function onTestEnd(array $logs=null) : void
    {
        if($this->callbacks==null)
            return;
        foreach ($this->callbacks as $callback) {
            $callback->onTestEnd($logs);
        }
    }

    public function onPredictEnd(array $logs=null) : void
    {
        if($this->callbacks==null)
            return;
        foreach ($this->callbacks as $callback) {
            $callback->onPredictEnd($logs);
        }
    }

    public function onTrainBatchBegin(int $batch, array $logs=null) : void
    {
        if($this->callbacks==null)
            return;
        foreach ($this->callbacks as $callback) {
            $callback->onTrainBatchBegin($batch, $logs);
        }
    }

    public function onTestBatchBegin(int $batch, array $logs=null) : void
    {
        if($this->callbacks==null)
            return;
        foreach ($this->callbacks as $callback) {
            $callback->onTestBatchBegin($batch, $logs);
        }
    }

    public function onTrainBatchEnd(int $batch, array $logs=null) : void
    {
        if($this->callbacks==null)
            return;
        foreach ($this->callbacks as $callback) {
            $callback->onTrainBatchEnd($batch, $logs);
        }
    }

    public function onTestBatchEnd(int $batch, array $logs=null) : void
    {
        if($this->callbacks==null)
            return;
        foreach ($this->callbacks as $callback) {
            $callback->onTrainBatchEnd($batch, $logs);
        }
    }

    public function onEpochBegin(int $epoch, array $logs=null) : void
    {
        if($this->callbacks==null)
            return;
        foreach ($this->callbacks as $callback) {
            $callback->onEpochBegin($epoch, $logs);
        }
    }

    public function onEpochEnd(int $epoch, array $logs=null) : void
    {
        if($this->callbacks==null)
            return;
        foreach ($this->callbacks as $callback) {
            $callback->onEpochEnd($epoch, $logs);
        }
    }
}
