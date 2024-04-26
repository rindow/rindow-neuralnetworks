<?php
namespace Rindow\NeuralNetworks\Callback;

use Rindow\NeuralNetworks\Model\Model;
use InvalidArgumentException;

/**
 *
 */
class CallbackList implements Broadcaster
{
    /** @var array<mixed> $callbacks */
    protected ?array $callbacks=null;
    protected Model $model;

    /**
     * @param array<mixed> $callbacks
     */
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

    public function onTrainBegin(array $metrics=null) : void
    {
        if($this->callbacks==null)
            return;
        foreach ($this->callbacks as $callback) {
            $callback->onTrainBegin($metrics);
        }
    }

    public function onTestBegin(array $metrics=null) : void
    {
        if($this->callbacks==null)
            return;
        foreach ($this->callbacks as $callback) {
            $callback->onTestBegin($metrics);
        }
    }

    public function onPredictBegin(array $metrics=null) : void
    {
        if($this->callbacks==null)
            return;
        foreach ($this->callbacks as $callback) {
            $callback->onPredictBegin($metrics);
        }
    }

    public function onTrainEnd(array $metrics=null) : void
    {
        if($this->callbacks==null)
            return;
        foreach ($this->callbacks as $callback) {
            $callback->onTrainEnd($metrics);
        }
    }

    public function onTestEnd(array $metrics=null) : void
    {
        if($this->callbacks==null)
            return;
        foreach ($this->callbacks as $callback) {
            $callback->onTestEnd($metrics);
        }
    }

    public function onPredictEnd(array $metrics=null) : void
    {
        if($this->callbacks==null)
            return;
        foreach ($this->callbacks as $callback) {
            $callback->onPredictEnd($metrics);
        }
    }

    public function onTrainBatchBegin(int $batch, array $metrics=null) : void
    {
        if($this->callbacks==null)
            return;
        foreach ($this->callbacks as $callback) {
            $callback->onTrainBatchBegin($batch, $metrics);
        }
    }

    public function onTestBatchBegin(int $batch, array $metrics=null) : void
    {
        if($this->callbacks==null)
            return;
        foreach ($this->callbacks as $callback) {
            $callback->onTestBatchBegin($batch, $metrics);
        }
    }

    public function onTrainBatchEnd(int $batch, array $metrics=null) : void
    {
        if($this->callbacks==null)
            return;
        foreach ($this->callbacks as $callback) {
            $callback->onTrainBatchEnd($batch, $metrics);
        }
    }

    public function onTestBatchEnd(int $batch, array $metrics=null) : void
    {
        if($this->callbacks==null)
            return;
        foreach ($this->callbacks as $callback) {
            $callback->onTrainBatchEnd($batch, $metrics);
        }
    }

    public function onEpochBegin(int $epoch, array $metrics=null) : void
    {
        if($this->callbacks==null)
            return;
        foreach ($this->callbacks as $callback) {
            $callback->onEpochBegin($epoch, $metrics);
        }
    }

    public function onEpochEnd(int $epoch, array $metrics=null) : void
    {
        if($this->callbacks==null)
            return;
        foreach ($this->callbacks as $callback) {
            $callback->onEpochEnd($epoch, $metrics);
        }
    }
}
