<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class Debug extends AbstractLayer
{
    use GenericUtils;

    protected bool $enable;
    protected mixed $forwardHook;
    protected mixed $backwardHook;

    /**
     * @param array<int> $input_shape
     */
    public function __construct(
        object $backend,
        ?array $input_shape=null,
        ?bool $enable=null,
        ?callable $forward_hook=null,
        ?callable $backward_hook=null,
        ?string $name=null,
    )
    {
        $enable ??= false;

        parent::__construct($backend);
        $this->inputShape = $input_shape;
        $this->enable = $enable;
        $this->forwardHook = $forward_hook;
        $this->backwardHook = $backward_hook;
        $this->initName($name,'debug');
    }

    public function build(mixed $variable=null, ?array $sampleWeights=null) : void
    {
        $K = $this->backend;

        $inputShape = $this->normalizeInputShape($variable);
        $this->outputShape = $inputShape;
    }

    public function getParams() : array
    {
        return [];
    }

    public function getGrads() : array
    {
        return [];
    }

    public function getConfig() : array
    {
        return [
            'options' => [
                'input_shape'=>$this->inputShape,
            ]
        ];
    }

    protected function call(NDArray $inputs, ?bool $training=null) : NDArray
    {
        $K = $this->backend;
        if($this->enable && $this->forwardHook) {
            $forwardHook = $this->forwardHook;
            $inputs = $forwardHook($inputs);
        }
        return $inputs;
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        if($this->enable && $this->backwardHook) {
            $backwardHook = $this->backwardHook;
            $dOutputs = $backwardHook($dOutputs);
        }
        return $dOutputs;
    }
}
