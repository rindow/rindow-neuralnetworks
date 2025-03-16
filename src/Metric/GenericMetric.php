<?php
namespace Rindow\NeuralNetworks\Metric;

use Interop\Polite\Math\Matrix\NDArray;

class GenericMetric extends AbstractMetric
{
    protected string $name = 'generic_metric';
    /** @var string|object|array{object,string} $func */
    protected string|object|array $func;

    public function __construct(
        object $backend,
        callable $func,
        ?string $name=null,
        )
    {
        parent::__construct($backend);
        $this->func = $func;
        if($name!==null) {
            $this->name = $name;
        }
    }

    public function forward(NDArray $trues, NDArray $predicts) : float
    {
        return ($this->func)($trues, $predicts);
    }
}
