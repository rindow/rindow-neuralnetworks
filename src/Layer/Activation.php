<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class Activation extends AbstractLayer
{
    use GenericUtils;
    protected $backend;

    public function __construct(
        object $backend,
        string|object $activation,
        array $input_shape=null,
        string $name=null,
    )
    {
        // defaults
        $input_shape = $input_shape ?? null;
        $name = $name ?? null;

        $this->backend = $K = $backend;
        $this->inputShape = $input_shape;
        $this->initName($name,'activation');
        $this->setActivation($activation);
    }

    public function getConfig() : array
    {
        return [
            'activation'=>$this->activationName,
            'input_shape'=>$this->inputShape,
        ];
    }

    protected function call(NDArray $inputs, bool $training) : NDArray
    {
        $outputs = $inputs;
        if($this->activation) {
            $container = $this->container();
            $outputs = $this->activation->forward($container,$outputs,$training);
        }
        return $outputs;
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $dInputs = $dOutputs;
        if($this->activation) {
            $container = $this->container();
            $dInputs = $this->activation->backward($container,$dOutputs);
        }
        return $dInputs;
    }
}
