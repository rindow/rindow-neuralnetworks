<?php
namespace Rindow\NeuralNetworks\Optimizer;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Optimizer\Schedule\LearningRateSchedule;
use Rindow\NeuralNetworks\Optimizer\Schedule\InverseTimeDecay;

class RMSprop implements Optimizer
{
    protected object $backend;
    protected float|LearningRateSchedule $lr;
    protected float $rho;
    protected float $decay;
    protected NDArray $iter;
    /** @var array<NDArray> $a */
    protected ?array $a=null;
    protected float $epsilon;

    public function __construct(
        object $backend,
        float|LearningRateSchedule|null $lr=null,
        ?float $rho=null,
        ?float $decay=null,
        ?float $epsilon=null,
    )
    {
        // defaults
        $lr = $lr ?? 0.001;
        $rho = $rho ?? 0.9;
        $decay = $decay ?? 0.0;
        $epsilon = $epsilon ?? null;

        $this->backend = $K = $backend;
        $this->lr = $lr;
        $this->rho = $rho;
        $this->decay = $decay;
        if(is_numeric($lr) && $decay!=0.0) {
            $this->lr = new InverseTimeDecay($lr,$decaySteps=1,$decay);
        }

        if($epsilon===null) {
            $epsilon = $K->epsilon();
        }
        $this->epsilon = $epsilon;
    }

    /**
     * @return array<NDArray>
     */
    public function getWeights() : array
    {
        if($this->a === null) {
            return [];
        }

        return array_merge([$this->iter],$this->a);
    }

    /**
     * @param array<NDArray> $params
     */
    public function loadWeights(array $params) : void
    {
        $this->iter = array_shift($params);
        $this->a = $params;
    }

    /**
     * @return array<string,mixed>
     */
    public function getConfig() : array
    {
        return [
            'options' => [
                'lr'      => $this->lr,
                'rho'     => $this->rho,
                'decay'   => $this->decay,
                'epsilon' => $this->epsilon,
            ],
        ];
    }

    /**
     * @param array<NDArray|Variable> $params
     */
    public function build(array $params) : void
    {
        $K = $this->backend;
        $this->a = array_map(fn($p)=>$K->zerosLike($p),$params);
        $this->iter = $K->zeros([]);
    }

    /**
     * @param array<NDArray|Variable> $params
     * @return array<NDArray>
     */
    protected function extractVariable(array $params)
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
        if($this->a === null) {
            $this->build($params);
        }

        $K->update_increment($this->iter,1.0);
        $iter = $this->iter->toArray();
        $lr = $this->learningRate((int)floor($iter));
        //if($this->decay > 0) {
        //    $lr = $lr * (1 / (1 + $this->decay * $iter));
        //}

        foreach(array_map(null, $params, $grads, $this->a) as [$p, $g, $a]) {
            # update accumulator

            // new_a = rho*a + (1-rho)*(g^2)
            $new_a = $K->add( $K->scale($this->rho,$a), $K->scale((1-$this->rho),$K->square($g)));

            // p -= lr*g / sqrt(new_a+epsilon)
            $K->update_sub($p,$K->mul($g, $K->rsqrt($new_a,$this->epsilon)), $lr);
        }
    }

    public function __clone()
    {
        if($this->a!=null) {
            $a = [];
            foreach ($this->a as $key => $value) {
                $a[] = clone $value;
            }
            $this->a = $a;
        }
        if(isset($this->iter)) {
            $this->iter = clone $this->iter;
        }
    }
}
