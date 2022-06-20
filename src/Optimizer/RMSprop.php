<?php
namespace Rindow\NeuralNetworks\Optimizer;

use Rindow\NeuralNetworks\Support\GenericUtils;

class SGD implements Optimizer
{
    use GenericUtils;
    protected $backend;
    protected $lr;

    public function __construct($backend, array $options=null)
    {
        extract($this->extractArgs([
            'lr'=>0.01,
        ],$options));
        $this->backend = $K = $backend;
        $this->lr = $lr;
    }

    public function getWeights() : array
    {
        return [
        ];
    }

    public function getConfig() : array
    {
        return [
            'options' => [
                'lr'      => $this->lr,
            ],
        ];
    }

    public function build(array $params) : void
    {
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
        foreach(array_keys($params) as $key) {
            $p = $params[$key];
            if($p instanceof Variable) {
                $p = $p->value();
            }
            // PARAM -=  lr * GRAD
            $K->update_sub($p,$K->scale($this->lr,$grads[$key]));
        }
    }
}
