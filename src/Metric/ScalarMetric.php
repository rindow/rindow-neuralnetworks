<?php
namespace Rindow\NeuralNetworks\Metric;

use Interop\Polite\Math\Matrix\NDArray;
use LogicException;

class ScalarMetric extends AbstractMetric
{
    protected string $name;

    public function __construct(
        object $backend,
        string $name,
        )
    {
        parent::__construct($backend);
        $this->name = $name;
    }

    public function forward(NDArray $trues, NDArray $predicts) : float
    {
        throw new LogicException('unsupported operation: "forward"');
    }

    public function update(NDArray $trues, NDArray $predicts) : void
    {
        throw new LogicException('unsupported operation: "update"');
    }
}
