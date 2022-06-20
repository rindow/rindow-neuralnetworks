<?php
namespace Rindow\NeuralNetworks\Loss;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;
use InvalidArgumentException;
use DomainException;

class Huber extends AbstractGradient implements Loss
{
    use GenericUtils;
    protected $backend;
    protected $delta;
    protected $trues;
    protected $predicts;

    public function __construct($backend,array $options=null)
    {
        extract($this->extractArgs([
            'delta'=>1.0,
        ],$options));
        $this->backend = $K = $backend;
        $this->delta = $delta;
    }

    public function getConfig() : array
    {
        return [
        ];
    }

    public function forward(NDArray $trues, NDArray $predicts) : float
    {
        $K = $this->backend;
        //$this->assertOutputShape($predicts);
        //if($trues->ndim()!=2) {
        //    throw new InvalidArgumentException('categorical\'s "trues" must be shape of [batchsize,1].');
        //}
        if($trues->shape()!=$predicts->shape())
            throw new InvalidArgumentException('unmatch shape of trues and predicts results');
        $N = $trues->size();
        #  x = trues - predicts
        #  loss = 0.5 * x^2                  if |x| <= d
        #  loss = 0.5 * d^2 + d * (|x| - d)  if |x| > d
        #       = d*|x| - 0.5*d^2
        #       = d*(|x| - 0.5*d)
        $this->trues = $trues;
        $this->predicts = $predicts;
        $x = $K->sub($trues,$predicts);
        $absx = $K->abs($x);
        $lessThenDelta = $K->lessEqual($absx, $this->delta);
        $greaterThenDelta = $K->greater($absx, $this->delta);
        $squaredLoss = $K->scale(0.5, $K->square($x));
        $linearLoss = $K->scale($this->delta, $K->increment($absx, -0.5*$this->delta));
        $loss = $K->add(
            $K->mul($lessThenDelta,   $squaredLoss),
            $K->mul($greaterThenDelta,$linearLoss)
        );
        $this->diffx = $x;
        $this->lessThenDelta = $lessThenDelta;
        $this->greaterThenDelta = $greaterThenDelta;
        return $K->scalar($K->sum($loss))/$N;
    }

    public function backward(array $dOutputs) : array
    {
        $K = $this->backend;
        $x = $this->diffx;
        $n = $x->size();
        $dSquaredLoss = $K->scale(-1/$n,$x);
        $dLinearLoss = $K->scale(-$this->delta/$n,$K->sign($x));
        $dInputs = $K->add(
            $K->mul($this->lessThenDelta,   $dSquaredLoss),
            $K->mul($this->greaterThenDelta,$dLinearLoss)
        );
        return [$dInputs];
    }

    public function accuracy(
        NDArray $trues, NDArray $predicts) : float
    {
        $K = $this->backend;
        if($trues->shape()!=$predicts->shape())
            throw new InvalidArgumentException('unmatch shape of trues and predicts results');
        // calc accuracy
        $shape=$predicts->shape();
        if(count($shape)>=2) {
            if($shape[1]>2147483648) {
                $dtype = NDArray::int64;
            } else {
                $dtype = NDArray::int32;
            }
            $predicts = $K->argmax($predicts, $axis=1,$dtype);
            $trues = $K->argmax($trues, $axis=1,$dtype);
            $sum = $K->sum($K->equal($trues, $predicts));
        } else {
            $sum = $K->nrm2($K->sub($predicts,$trues));
        }
        $sum = $K->scalar($sum);
        $accuracy = $sum / (float)$trues->shape()[0];
        return $accuracy;
    }
}
