<?php
namespace Rindow\NeuralNetworks\Optimizer;

use Rindow\NeuralNetworks\Gradient\Variable;
use UnexpectedValueException;
use Rindow\NeuralNetworks\Optimizer\Schedule\LearningRateSchedule;

class Adam implements Optimizer
{
    protected $backend;
    protected $lr;
    protected $beta1;
    protected $beta2;
    protected $iter;
    protected $m;
    protected $v;
    protected $epsilon;

    public function __construct(
        object $backend,
        float|LearningRateSchedule $lr=null,
        float $beta1=null,
        float $beta2=null,
        float $epsilon=null,
    )
    {
        // defaults
        $lr      = $lr ?? 0.001;
        $beta1   = $beta1 ?? 0.9;
        $beta2   = $beta2 ?? 0.999;
        $epsilon = $epsilon ?? null;

        $this->backend = $K = $backend;
        $this->lr = $lr;
        $this->beta1 = $beta1;
        $this->beta2 = $beta2;
        if($epsilon===null) {
            $epsilon = $K->epsilon();
        }
        $this->epsilon = $epsilon;
    }

    public function getWeights() : array
    {
        if($this->m === null) {
            return [];
        }
        return array_merge([$this->iter],$this->m,$this->v);
    }

    public function loadWeights(array $params) : void
    {
        $this->iter = array_shift($params);
        $count = (int)intval(count($params)/2);
        $m = [];
        for($i=0;$i<$count;$i++) {
            $m[] = array_shift($params);
        }
        $this->m = $m;
        $this->v = $params;
    }

    public function getConfig() : array
    {
        return [
            'options' => [
                'lr'      => $this->lr,
                'beta1'   => $this->beta1,
                'beta2'   => $this->beta2,
                'epsilon' => $this->epsilon,
            ],
        ];
    }

    public function build(array $params) : void
    {
        $K = $this->backend;
        foreach ($params as $key => $value) {
            $this->m[$key] = $K->zerosLike($value);
            $this->v[$key] = $K->zerosLike($value);
        }
        $this->iter = $K->zeros([]);
    }

    protected function extractVariable($params)
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

    public function learningRate(mixed $step) : float
    {
        $lr = $this->lr;
        if(is_numeric($lr)) {
            // lr_t = lr * sqrt( 1 - beta_2**t ) /
            //                 ( 1 - beta_1**t )
            $lr_t = $lr * sqrt(1.0 - ($this->beta2**$step)) /
                                    (1.0 - ($this->beta1**$step)) ;
            return $lr_t;
        }
        return $lr($step);
    }

    public function update(array $params, array $grads) : void
    {
        $K = $this->backend;
        $params = $this->extractVariable($params);
        if($this->m === null) {
            $this->build($params);
        }

        $K->update_increment($this->iter,1.0);
        $iter = $this->iter->toArray();

        $lr_t = $this->learningRate($iter);

        foreach(array_map(null,$params,$grads,$this->m,$this->v) as [$p,$g,$m,$v]) {
            // m += ( 1 - beta_1 ) * ( g - m )
            // v += ( 1 - beta_2 ) * ( g**2 - v )
            // p -= lr_t * m / ( sqrt(v) + epsilon )
            $K->update_add($m, $K->sub($g, $m), (1 - $this->beta1));
            $K->update_add($v, $K->sub($K->square($g),$v), (1 - $this->beta2));
            $K->update_sub($p, $K->mul($m, $K->rsqrt($v,$this->epsilon)), $lr_t);
        }
    }

    public function __clone()
    {
        if($this->m!=null) {
            $m = [];
            foreach ($this->m as $key => $value) {
                $m[] = clone $value;
            }
            $this->m = $m;
        }
        if($this->v!=null) {
            $v = [];
            foreach ($this->v as $key => $value) {
                $v[] = clone $value;
            }
            $this->v = $v;
        }
        if($this->iter!=null) {
            $this->iter = clone $this->iter;
        }
    }
}
