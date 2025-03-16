<?php
namespace Rindow\NeuralNetworks\Callback;

interface Events
{
    /**
     * @param array<mixed> $metrics
     */
    public function onTrainBegin(?array $metrics=null) : void;

    /**
     * @param array<mixed> $metrics
     */
    public function onTestBegin(?array $metrics=null) : void;

    /**
     * @param array<mixed> $metrics
     */
    public function onPredictBegin(?array $metrics=null) : void;

    /**
     * @param array<mixed> $metrics
     */
    public function onTrainEnd(?array $metrics=null) : void;

    /**
     * @param array<mixed> $metrics
     */
    public function onTestEnd(?array $metrics=null) : void;

    /**
     * @param array<mixed> $metrics
     */
    public function onPredictEnd(?array $metrics=null) : void;

    /**
     * @param array<mixed> $metrics
     */
    public function onTrainBatchBegin(int $batch, ?array $metrics=null) : void;

    /**
     * @param array<mixed> $metrics
     */
    public function onTestBatchBegin(int $batch, ?array $metrics=null) : void;

    /**
     * @param array<mixed> $metrics
     */
    public function onTrainBatchEnd(int $batch, ?array $metrics=null) : void;

    /**
     * @param array<mixed> $metrics
     */
    public function onTestBatchEnd(int $batch, ?array $metrics=null) : void;

    /**
     * @param array<mixed> $metrics
     */

     public function onEpochBegin(int $epoch, ?array $metrics=null) : void;
    /**
     * @param array<mixed> $metrics
     */
    public function onEpochEnd(int $epoch, ?array $metrics=null) : void;
}
