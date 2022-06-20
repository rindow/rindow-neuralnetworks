<?php
namespace Rindow\NeuralNetworks\Optimizer;

use Rindow\NeuralNetworks\Support\GenericUtils;
use Rindow\NeuralNetworks\Gradient\Core\Variable;
use UnexpectedValueException;

class Adam implements Optimizer
{
    use GenericUtils;
    protected $backend;
    protected $lr;
    protected $beta1;
    protected $beta2;
    protected $iter;
    protected $m;
    protected $v;
    protected $epsilon;

    public function __construct($backend, array $options=null)
    {
        extract($this->extractArgs([
            'lr'      => 0.001,
            'beta1'   => 0.9,
            'beta2'   => 0.999,
            'epsilon' => null,
        ],$options));
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

    public function update(array $params, array $grads) : void
    {
        $K = $this->backend;
        $params = $this->extractVariable($params);
        if($this->m === null) {
            $this->build($params);
        }

        $K->update_increment($this->iter,1.0);
        $iter = $this->iter->toArray();
        // t = K.cast(self.iterations, K.floatx()) + 1
        // lr_t = lr * sqrt( 1 - beta_2**t ) /
        //                 ( 1 - beta_1**t )
        $lr_t = $this->lr * sqrt(1.0 - ($this->beta2**$iter)) /
                                (1.0 - ($this->beta1**$iter)) ;

        foreach (array_keys($params) as $key) {
            $p = $params[$key];
            $g = $grads[$key];
            $m = $this->m[$key];
            $v = $this->v[$key];

            // m = ( beta_1 * m ) + ( 1 - beta_1 ) * g
            // v = ( beta_2 * v ) + ( 1 - beta_2 ) * g**2
            // p = p - lr_t * m / ( sqrt(v) + epsilon )
            #$K->update($m,$K->add($K->scale($this->beta1,$m),
            #                      $K->scale(1.0-$this->beta1,$g)));
            #$K->update($v,$K->add($K->scale($this->beta2,$v),
            #                      $K->scale(1.0-$this->beta2,$K->square($g))));
            #$K->update($p,$K->sub($p,$K->mul($K->scale($lr_t,$m),
            #                                 $K->rsqrt($v,$this->epsilon))));

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
