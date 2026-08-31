<?php
namespace Rindow\NeuralNetworks\Optimizer;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Variable;
use UnexpectedValueException;
use Rindow\NeuralNetworks\Optimizer\Schedule\LearningRateSchedule;

class Adam implements Optimizer
{
    protected object $backend;
    protected float|LearningRateSchedule $lr;
    protected float $beta1;
    protected float $beta2;
    protected NDArray $iter;
    /** @var array<NDArray> $m */
    protected ?array $m=null;
    /** @var array<NDArray> $v */
    protected ?array $v=null;
    protected float $epsilon;

    public function __construct(
        object $backend,
        float|LearningRateSchedule|null $lr=null,
        ?float $beta1=null,
        ?float $beta2=null,
        ?float $epsilon=null,
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

    /**
     * @return array<NDArray>
     */
    public function getWeights() : array
    {
        if($this->m === null) {
            return [];
        }
        return array_merge([$this->iter],$this->m,$this->v);
    }

    /**
     * @param array<NDArray> $params
     */
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

    /**
     * @return array<string,mixed>
     */
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

    /**
     * @param array<NDArray|Variable> $params
     */
    public function build(array $params) : void
    {
        $K = $this->backend;
        foreach ($params as $key => $value) {
            $this->m[$key] = $K->zerosLike($value);
            $this->v[$key] = $K->zerosLike($value);
        }
        $this->iter = $K->zeros([]);
    }

    /**
     * @param array<NDArray|Variable> $params
     * @return array<NDArray>
     */
    protected function extractVariable(array $params) : array
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
/*
    public function learningRate(int $step) : float
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

        $lr_t = $this->learningRate((int)floor($iter));

        foreach(array_map(null,$params,$grads,$this->m,$this->v) as [$p,$g,$m,$v]) {
            // m += ( 1 - beta_1 ) * ( g - m )
            // v += ( 1 - beta_2 ) * ( g**2 - v )
            // p -= lr_t * m / ( sqrt(v) + epsilon )
            $K->update_add($m, $K->sub($g, $m), (1 - $this->beta1));
            $K->update_add($v, $K->sub($K->square($g),$v), (1 - $this->beta2));
            $K->update_sub($p, $K->mul($m, $K->rsqrt($v,$this->epsilon)), $lr_t);
        }
    }
*/
    public function learningRate(int $step) : float
    {
        if($this->lr instanceof LearningRateSchedule) {
            return ($this->lr)($step);
        }
        return $this->lr;
    }

    public function update(array $params, array $grads) : void
    {
        $K = $this->backend;
        $params = $this->extractVariable($params);
        if($this->m === null) {
            $this->build($params);
        }

        $K->update_increment($this->iter, 1.0);
        $t = $this->iter->toArray(); // ステップ数 t
        $t = (int)floor($t);

        // 現在のステップでの学習率を取得
        $lr = $this->learningRate($t);

        // バイアス補正項を計算
        $beta1_t = $this->beta1 ** $t;
        $beta2_t = $this->beta2 ** $t;
        $bias_correction1 = 1.0 - $beta1_t;
        $bias_correction2 = 1.0 - $beta2_t;
        
        // Pytorchでは小さな値による除算を避けるための安全策があるが、
        // まずは基本的な実装で試す
        if ($bias_correction1 == 0.0) {
            throw new \RuntimeException("bias_correction1 is zero.");
        }
        if ($bias_correction2 == 0.0) {
            throw new \RuntimeException("bias_correction2 is zero.");
        }

        foreach(array_map(null, $params, $grads, $this->m, $this->v) as [$p, $g, $m, $v]) {
            // 1. モーメントの更新
            // m = beta1 * m + (1 - beta1) * g
            // v = beta2 * v + (1 - beta2) * g^2
            $K->update($m, $K->add($K->scale($this->beta1, $m), $K->scale(1.0 - $this->beta1, $g)));
            $K->update($v, $K->add($K->scale($this->beta2, $v), $K->scale(1.0 - $this->beta2, $K->square($g))));

            // 2. バイアス補正されたモーメントを計算 (m_hat, v_hat)
            // m_hat = m / (1 - beta1^t)
            $m_hat = $K->scale(1.0 / $bias_correction1, $m);
            // v_hat = v / (1 - beta2^t)
            $v_hat = $K->scale(1.0 / $bias_correction2, $v);
            
            // 3. パラメータの更新
            // p -= lr * m_hat / (sqrt(v_hat) + epsilon)
            $K->update_sub($p, $K->mul($m_hat, $K->rsqrt($v_hat, $this->epsilon)), $lr);
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
