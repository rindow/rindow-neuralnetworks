<?php
namespace Rindow\NeuralNetworks\Optimizer;

use Rindow\NeuralNetworks\Support\GenericUtils;

class RMSprop implements Optimizer
{
    use GenericUtils;
    protected $backend;
    protected $lr;
    protected $rho;
    protected $decay;
    protected $iter;
    protected $a;
    protected $epsilon;

    public function __construct($backend, array $options=null)
    {
        extract($this->extractArgs([
            'lr'=>0.001,
            'rho'=>0.9,
            'decay'=>0.0,
            'epsilon'=>null,
        ],$options));
        $this->backend = $K = $backend;
        $this->lr = $lr;
        $this->rho = $rho;
        $this->decay = $decay;
        if($epsilon===null) {
            $epsilon = $K->epsilon();
        }
        $this->epsilon = $epsilon;
    }

    public function getWeights() : array
    {
        if($this->a === null) {
            return [];
        }

        return array_merge([$this->iter],$this->a);
    }

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

    public function build(array $params) : void
    {
        $K = $this->backend;
        $this->a = array_map(function($p) use ($K) {return $K->zerosLike($p);},$params);
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

    public function update(array $params, array $grads) : void
    {
        $K = $this->backend;
        $params = $this->extractVariable($params);
        if($this->a === null) {
            $this->build($params);
        }

        $K->update_increment($this->iter,1.0);
        $iter = $this->iter->toArray();
        $lr = $this->lr;
        if($this->decay > 0) {
            $lr = $lr * (1 / (1 + $this->decay * $iter));
        }

        foreach(array_map(null, $params, $grads, $this->a) as $data) {
            [$p, $g, $a] = $data;
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
        if($this->iter!=null) {
            $this->iter = clone $this->iter;
        }
    }
}
