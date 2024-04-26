<?php
namespace Rindow\NeuralNetworks\Loss;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use ArrayAccess;

class CategoricalCrossEntropy extends AbstractLoss
{
    protected function call(NDArray $trues, NDArray $predicts) : NDArray
    {
        $K = $this->backend;
        if($this->fromLogits) {
            $predicts = $K->softmax($predicts);
        }
        [$trues,$predicts] = $this->flattenShapes($trues,$predicts);
        $container = $this->container();
        $container->trues = $trues;
        $container->predicts = $predicts;
        $outputs = $K->categoricalCrossEntropy(
            $trues, $predicts,
            $this->fromLogits, $this->reduction
        );
        $outputs = $this->reshapeLoss($outputs);
        return $outputs;
    }

    protected function differentiate(array $dOutputs, ArrayAccess $grads=null, array $oidsToCollect=null) : array
    {
        $K = $this->backend;
        $dLoss = $this->flattenLoss($dOutputs[0]);
        $container = $this->container();
        $dInputs = $K->dCategoricalCrossEntropy(
            $dLoss, $container->trues, $container->predicts,
            $this->fromLogits, $this->reduction
        );
        $dInputs = $this->reshapePredicts($dInputs);
        return [$dInputs];
    }

    public function accuracyMetric() : string
    {
        return 'categorical_accuracy';
    }
}
