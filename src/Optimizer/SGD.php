<?php
namespace Rindow\NeuralNetworks\Optimizer;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Optimizer\Schedule\LearningRateSchedule;


class SGD implements Optimizer
{
    protected object $backend;
    protected float|LearningRateSchedule $lr;

    public function __construct(
        object $backend,
        float|LearningRateSchedule|null $lr=null,
        )
    {
        // defaults
        $lr = $lr ?? 0.01;
        
        $this->backend = $K = $backend;
        $this->lr = $lr;
    }

    /**
     * @return array<NDArray>
     */
    public function getWeights() : array
    {
        return [
        ];
    }

    /**
     * @param array<NDArray> $params
     */
    public function loadWeights(array $params) : void
    {
    }

    /**
     * @return array<string,mixed>
     */
    public function getConfig() : array
    {
        return [
            'options' => [
                'lr'      => $this->lr,
            ],
        ];
    }

    /**
     * @param array<NDArray|Variable> $params
     */
    public function build(array $params) : void
    {
    }

    /**
     * @param array<NDArray|Variable> $params
     * @return array<NDArray>
     */
    protected function extractVariable($params) : array
    {
        $params2 = [];
        foreach($params as $p) {
            if($p instanceof Variable) {
                $p = $p->value();
            }
            $params2[] = $p;
        }
        return $params2;
    }

    protected function learningRate(int $step) : float
    {
        $lr = $this->lr;
        if(is_numeric($lr)) {
            return $lr;
        }
        return $lr($step);
    }

    /**
     * @param array<NDArray|Variable> $params
     * @param array<NDArray|Variable> $grads
     */
    public function update(array $params, array $grads) : void
    {
        $K = $this->backend;
        $params = $this->extractVariable($params);
        $grads = $this->extractVariable($grads);

        $lr = $this->learningRate(0);
        foreach(array_map(null,$params,$grads) as [$param,$grad]) {
            // PARAM -=  lr * GRAD
            $K->update_sub($param,$K->scale($lr,$grad));
        }
    }
}
