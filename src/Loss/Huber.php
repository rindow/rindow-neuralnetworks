<?php
namespace Rindow\NeuralNetworks\Loss;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use DomainException;
use ArrayAccess;

class Huber extends AbstractLoss
{
    protected float $delta;

    public function __construct(
        object $backend,
        float $delta=null,
        string $reduction=null,
    )
    {
        parent::__construct($backend,from_logits:null,reduction:$reduction);
        // defaults
        $delta = $delta ?? 1.0;
        $this->delta = $delta;
    }

    protected function call(NDArray $trues, NDArray $predicts) : NDArray
    {
        $K = $this->backend;
        [$trues,$predicts] = $this->flattenShapes($trues,$predicts);
        $container = $this->container();
        //$this->assertOutputShape($predicts);
        //if($trues->ndim()!=2) {
        //    throw new InvalidArgumentException('categorical\'s "trues" must be shape of [batchsize,1].');
        //}
        if($trues->shape()!=$predicts->shape())
            throw new InvalidArgumentException('unmatch shape of trues and predicts results');
        #  x = trues - predicts
        #  loss = 0.5 * x^2                  if |x| <= d
        #  loss = 0.5 * d^2 + d * (|x| - d)  if |x| > d
        #       = d*|x| - 0.5*d^2
        #       = d*(|x| - 0.5*d)
        $x = $K->sub($trues,$predicts);
        $absx = $K->abs($x);
        $lessThenDelta = $K->lessEqual($absx, $this->delta);
        $greaterThenDelta = $K->increment($K->copy($lessThenDelta),1.0,-1.0);
        $squaredLoss = $K->scale(0.5, $K->square($x));
        $linearLoss = $K->scale($this->delta, $K->increment($absx, -0.5*$this->delta));
        $loss = $K->add(
            $K->mul($lessThenDelta,   $squaredLoss),
            $K->mul($greaterThenDelta,$linearLoss),
        );
        $container->diffx = $x;
        $container->lessThenDelta = $lessThenDelta;
        $container->greaterThenDelta = $greaterThenDelta;
        if($this->reduction!='none') {
            $n = $loss->size();
            $loss = $K->sum($loss);
            if(is_numeric($loss)) {
                $loss = $K->array($loss,dtype:$predicts->dtype());
            }
            $K->update_scale($loss,1/$n);
        } else {
            $shape = $loss->shape();
            $n = array_pop($shape);
            $loss = $K->sum($loss,axis:-1);
            $K->update_scale($loss,1/$n);
        }
        $loss = $this->reshapeLoss($loss);
        return $loss;
    }

    protected function differentiate(array $dOutputs, ArrayAccess $grads=null, array $oidsToCollect=null) : array
    {
        $K = $this->backend;
        $dLoss = $this->flattenLoss($dOutputs[0]);
        $container = $this->container();
        $x = $K->copy($container->diffx);
        $dSquaredLoss = $x;
        $dLinearLoss = $K->sign($x);
        $K->update_scale($dLinearLoss,$this->delta);
        $dInputs = $K->add(
            $K->mul($container->lessThenDelta,   $dSquaredLoss),
            $K->mul($container->greaterThenDelta,$dLinearLoss),
        );
        if($this->reduction!='none') {
            $n = $x->size();
            $K->update_scale($dInputs,-1/$n);
        } else {
            $shape = $x->shape();
            $n = array_pop($shape);
            $K->update_scale($dInputs,-1/$n);
        }
        $trans = false;
        if($this->reduction=='none') {
            $trans = true;
        }
        $K->update_mul($dInputs,$dLoss,trans:$trans);
        $dInputs = $this->reshapePredicts($dInputs);
        return [$dInputs];
    }

    public function accuracyMetric() : string
    {
        return 'categorical_accuracy';
    }
}
