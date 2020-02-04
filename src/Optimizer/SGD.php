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

    public function update(array $params, array $grads) : void
    {
        $K = $this->backend;
        foreach(array_keys($params) as $key) {
            // PARAM -=  lr * GRAD
            $K->update_sub($params[$key],$K->scale($this->lr,$grads[$key]));
        }
    }
}
