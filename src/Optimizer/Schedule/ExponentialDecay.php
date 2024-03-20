<?php
namespace Rindow\NeuralNetworks\Optimizer\Schedule;

class ExponentialDecay implements LearningRateSchedule
{
    protected float $initialLearningRate;
    protected int $decaySteps;
    protected float $decayRate;
    protected bool $staircase;

    public function __construct(
        float $initialLearningRate,
        int $decaySteps,
        float $decayRate,
        bool $staircase=null,
        )
    {
        $staircase = $staircase ?? false;

        $this->initialLearningRate = $initialLearningRate;
        $this->decaySteps = $decaySteps;
        $this->decayRate = $decayRate;
        $this->staircase = $staircase;
    }

    public function __invoke(mixed $step) : float
    {
        $progress = ($step / $this->decaySteps);
        if($this->staircase) {
            $progress = floor($progress);
        }
        return $this->initialLearningRate * $this->decayRate**$progress ;
    }
}
