<?php
namespace Rindow\NeuralNetworks\Optimizer\Schedule;

class  InverseTimeDecay implements LearningRateSchedule
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

    public function __invoke(int $step) : float
    {
        $progress = ($step / $this->decaySteps);
        if($this->staircase) {
            $progress = floor($progress);
        }

        // lr = initial / ( 1 + decay_rate * step )
        return $this->initialLearningRate / (1 + $this->decayRate*$progress) ;
    }
}
