<?php
namespace Rindow\NeuralNetworks\Optimizer;

use Rindow\NeuralNetworks\Gradient\Variable;

class SGD implements Optimizer
{
    protected $backend;
    protected $lr;

    public function __construct(
        object $backend,
        float $lr=null,
        )
    {
        // defaults
        $lr = $lr ?? 0.01;
        
        $this->backend = $K = $backend;
        $this->lr = $lr;
    }

    public function getWeights() : array
    {
        return [
        ];
    }

    public function loadWeights(array $params) : void
    {
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
        $grads = $this->extractVariable($grads);
        foreach(array_map(null,$params,$grads) as [$param,$grad]) {
            // PARAM -=  lr * GRAD
            $K->update_sub($param,$K->scale($this->lr,$grad));
        }
    }
}
