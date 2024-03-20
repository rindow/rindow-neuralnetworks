<?php
namespace Rindow\NeuralNetworks\Optimizer\Schedule;

interface LearningRateSchedule
{
    public function __invoke(int $step) : float;
}
