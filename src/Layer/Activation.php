<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class Activation extends AbstractLayer
{
    use GenericUtils;

    /**
     * @param array<int> $input_shape
     */
    public function __construct(
        object $backend,
        string|object $activation,
        ?array $input_shape=null,
        ?string $name=null,
    )
    {
        // defaults
        $input_shape = $input_shape ?? null;
        $name = $name ?? null;

        parent::__construct($backend);
        $K = $backend;
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

    protected function call(
        NDArray $inputs,
        ?bool $training=null,
        mixed ...$kargs
    ) : NDArray
    {
        $outputs = $inputs;
        if($this->activation) {
            $container = $this->container();
            $outputs = $this->activation->forward($container,$outputs,$training, ...$kargs);
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
