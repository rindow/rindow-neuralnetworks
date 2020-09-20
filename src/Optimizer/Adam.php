<?php
namespace Rindow\NeuralNetworks\Optimizer;

use Rindow\NeuralNetworks\Support\GenericUtils;
use UnexpectedValueException;

class Adam implements Optimizer
{
    use GenericUtils;
    protected $backend;
    protected $lr;
    protected $beta1;
    protected $beta2;
    protected $iter = 0;
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

        return array_merge($this->m,$this->v);
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
    }

    public function update(array $params, array $grads) : void
    {
        $K = $this->backend;
        if($this->m === null) {
            $this->build($params);
        }

        $this->iter++;

        // t = K.cast(self.iterations, K.floatx()) + 1
        // lr_t = lr * sqrt( 1 - beta_2**t ) /
        //                 ( 1 - beta_1**t )
        $lr_t = $this->lr * sqrt(1.0 - ($this->beta2**$this->iter)) /
                                (1.0 - ($this->beta1**$this->iter)) ;

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
}
