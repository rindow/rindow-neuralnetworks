<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;

class Transpose extends AbstractFunction
{
    /** @var array<int> $perm */
    protected ?array $perm=null;
    /** @var array<int> $dPerm */
    protected ?array $dPerm=null;

    /** @param array<int> $perm */
    public function __construct(
        object $backend,
        array $perm=null,
    )
    {
        parent::__construct($backend);
        $this->perm = $perm;
        if($perm===null) {
            return;
        }
        $dPerm = [];
        foreach($perm as $i => $v) {
            $dPerm[(int)$v] = $i;
        }
        ksort($dPerm);
        $this->dPerm = $dPerm;
    }

    protected function call(array $inputs) : array
    {
        $K = $this->backend;
        $output = $K->transpose($inputs[0],perm:$this->perm);
        return [$output];
    }

    protected function differentiate(array $dOutputs) : array
    {
        $K = $this->backend;
        $dInput = $K->transpose($dOutputs[0],perm:$this->dPerm);
        return [$dInput];
    }
}
