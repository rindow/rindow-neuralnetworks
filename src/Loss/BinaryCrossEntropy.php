<?php
namespace Rindow\NeuralNetworks\Loss;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use ArrayAccess;

class BinaryCrossEntropy extends AbstractLoss
{
    protected function call(NDArray $trues, NDArray $predicts) : NDArray
    {
        $K = $this->backend;
        if($this->fromLogits) {
            $predicts = $K->sigmoid($predicts);
        }
        [$trues,$predicts] = $this->flattenShapes($trues,$predicts);
        $container = $this->container();
        $container->trues = $trues;
        $container->predicts = $predicts;
        $outputs = $K->binaryCrossEntropy(
            $trues, $predicts,
            $this->fromLogits, $this->reduction
        );
        $outputs = $this->reshapeLoss($outputs);
        return $outputs;
    }

    protected function differentiate(array $dOutputs, ?ArrayAccess $grads=null, ?array $oidsToCollect=null) : array
    {
        $K = $this->backend;
        $dLoss = $this->flattenLoss($dOutputs[0]);
        $container = $this->container();
        $dInputs = $K->dBinaryCrossEntropy(
            $dLoss, $container->trues, $container->predicts,
            $this->fromLogits, $this->reduction
        );
        $dInputs = $this->reshapePredicts($dInputs);
        return [$dInputs];
    }

    public function accuracyMetric() : string
    {
        return 'binary_accuracy';
    }
}
