<?php
namespace Rindow\NeuralNetworks\Backend\RindowBlas;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

class Backend
{
    protected $initializers = [
        'relu_normal'       => 'initializer_relu',
        'sigmoid_normal'    => 'initializer_sigmoid',
        'zeros'             => 'zeros',
        'ones'              => 'ones',
    ];
    protected $epsilon = 1e-7;
    protected $equalEpsilon = 1e-06;
    protected $matrixOperator;
    protected $la;

    public function __construct($matrixOperator)
    {
        $this->matrixOperator = $matrixOperator;
        $this->la = $matrixOperator->laRawMode();
    }

    public function epsilon()
    {
        return $this->epsilon;
    }

    public function setEpsilon($epsilon) : void
    {
        $this->epsilon = $epsilon;
    }

    public function dtypeToString($dtype) : string
    {
        $mo = $this->matrixOperator;
        return $mo->dtypeToString($dtype);
    }

    public function alloc(array $shape,$dtype=null)
    {
        return $this->la->alloc($shape,$dtype);
    }

    public function allocLike(NDArray $x)
    {
        return $this->la->alloc($x->shape(),$x->dtype());
    }

    public function getInitializer($name)
    {
        if(!array_key_exists($name,$this->initializers))
            throw new InvalidArgumentException('Unsupported initializer: '.$name);
        return [$this,$this->initializers[$name]];
    }

    public function initializer_relu(array $shape)
    {
        $mo = $this->matrixOperator;
        $inputSize = $shape[0];
        $scale = sqrt(2.0 / $inputSize);
        $kernel = $this->la->scal($scale,$mo->random()->randn($shape));
        return $kernel;
    }

    public function initializer_sigmoid(array $shape)
    {
        $mo = $this->matrixOperator;
        $inputSize = $shape[0];
        $scale = sqrt(1.0 / $inputSize);
        $kernel = $this->la->scal($scale,$mo->random()->randn($shape));
        return $kernel;
    }

    public function zeros(array $shape)
    {
        $x = $this->la->alloc($shape);
        $this->la->zeros($x);
        return $x;
    }

    public function ones(array $shape)
    {
        $x = $this->la->alloc($shape);
        $this->la->zeros($x);
        $this->la->increment($x,1.0);
        return $x;
    }

    public function zerosLike(NDArray $x)
    {
        $y = $this->la->alloc($x->shape(),$x->dtype());
        $this->la->zeros($y);
        return $y;
    }

    public function onesLike(NDArray $x)
    {
        $y = $this->la->alloc($x->shape());
        $this->la->zeros($y);
        $this->la->increment($y,1.0);
        return $y;
    }

    public function copy(NDArray $from,NDArray $to=null)
    {
        $this->la->copy($from, $to);
        return $to;
    }

    public function cast(NDArray $x,$dtype)
    {
        $mo = $this->matrixOperator;
        return $mo->astype($x,$dtype);
    }

    public function transpose(NDArray $x)
    {
        $mo = $this->matrixOperator;
        return $mo->transpose($x);
    }

    public function add(NDArray $x, NDArray $y)
    {
        $ndimX = $x->ndim();
        $ndimY = $y->ndim();
        if($ndimX == $ndimY) {
            $x = $this->la->copy($x,$this->la->alloc($y->shape(),$x->dtype()));
            $this->la->axpy($y,$x);
            return $x;
        } elseif($ndimX < $ndimY) {
            $x = $this->la->duplicate($x,null,null,$this->la->alloc($y->shape(),$x->dtype()));
            $this->la->axpy($y,$x);
            return $x;
        } else {
            $y = $this->la->duplicate($y,null,null,$this->la->alloc($x->shape(),$y->dtype()));
            $this->la->axpy($x,$y);
            return $y;
        }
    }

    public function sub(NDArray $x, NDArray $y)
    {
        $ndimX = $x->ndim();
        $ndimY = $y->ndim();
        if($ndimX == $ndimY) {
            $x = $this->la->copy($x,$this->la->alloc($y->shape(),$x->dtype()));
            $this->la->axpy($y,$x,-1.0);
            return $x;
        } elseif($ndimX < $ndimY) {
            $x = $this->la->duplicate($x,null,null,$this->la->alloc($y->shape(),$x->dtype()));
            $this->la->axpy($y,$x,-1.0);
            return $x;
        } else {
            $y = $this->la->duplicate($y,null,null,$this->la->alloc($x->shape(),$y->dtype()));
            $this->la->increment($y,0,-1);
            $this->la->axpy($x,$y);
            return $y;
        }
    }

    public function mul(NDArray $x, NDArray $y)
    {
        if($x->ndim() < $y->ndim()) {
            $y = $this->la->copy($y);
            return $this->la->multiply($x,$y);
        } else {
            $x = $this->la->copy($x);
            return $this->la->multiply($y,$x);
        }
    }

    public function div(NDArray $x, NDArray $y)
    {
        $y = $this->la->copy($y);
        $this->la->reciprocal($y);
        if($x->ndim() < $y->ndim()) {
            return $this->la->multiply($x,$y);
        } else {
            $m = $this->la->copy($x);
            return $this->la->multiply($y,$x);
        }
    }

    public function update(NDArray $x, NDArray $newX) : NDArray
    {
        $this->la->copy($newX,$x);
        return $x;
    }

    public function update_add(NDArray $x, NDArray $increment) : NDArray
    {
        $this->la->axpy($increment,$x);
        return $x;
    }

    public function update_sub(NDArray $x, NDArray $decrement) : NDArray
    {
        $this->la->axpy($decrement,$x,-1.0);
        return $x;
    }

    public function update_mul(NDArray $x, NDArray $magnifications) : NDArray
    {
        return $this->la->multiply($magnifications,$x);
    }

    public function scale(float $a, NDArray $x)
    {
        $x = $this->la->copy($x);
        return $this->la->scal($a, $x);
    }

    public function increment(NDArray $x, float $a)
    {
        $x = $this->la->copy($x);
        return $this->la->increment($x, $a);
    }

    public function pow(NDArray $x, float $y)
    {
        $x = $this->la->copy($x);
        return $this->la->pow($x,$y);
    }

    public function square(NDArray $x)
    {
        $x = $this->la->copy($x);
        return $this->la->square($x);
    }

    public function sqrt(NDArray $x)
    {
        $x = $this->la->copy($x);
        return $this->la->sqrt($x);
    }

    public function rsqrt(NDArray $x,float $beta=null, float $alpha=null)
    {
        $x = $this->la->copy($x);
        return $this->la->rsqrt($x,$beta,$alpha);
    }

    public function maximum(NDArray $x, float $a)
    {
        $x = $this->la->copy($x);
        return $this->la->maximum($a,$x);
    }

    public function minimum(NDArray $x, float $a)
    {
        $x = $this->la->copy($x);
        return $this->la->minimum($a,$x);
    }

    public function dmaximum(NDArray $x, float $a)
    {
        $x = $this->la->copy($x);
        return $this->la->dmaximum($a,$x);
    }

    public function dminimum(NDArray $x, float $a)
    {
        $x = $this->la->copy($x);
        return $this->la->dminimum($a,$x);
    }

    public function exp(NDArray $x)
    {
        $x = $this->la->copy($x);
        return $this->la->exp($x);
    }

    public function log(NDArray $x)
    {
        $x = $this->la->copy($x);
        return $this->la->log($x);
    }

    public function greater($x,$y)
    {
        $mo = $this->matrixOperator;
        return $mo->op($x,'>',$y);
    }

    public function equal($x,$y)
    {
        return $this->la->equal($x,$y);
    }

    public function sum(NDArray $x,$axis=null)
    {
        if($axis===null) {
            return $this->la->sum($x);
        } else {
            return $this->la->reduceSum($x,$axis);
        }
    }

    public function mean(NDArray $x,int $axis=null)
    {
        if($axis===null) {
            return $this->la->sum($x) / $x->size();
        } else {
            return $this->la->reduceMean($x,$axis);
        }
    }

    public function max(NDArray $x,int $axis=null)
    {
        if($axis===null) {
            return $this->la->max($x);
        } else {
            return $this->la->reduceMax($x,$axis);
        }
    }

    public function min(NDArray $x,int $axis=null)
    {
        $mo = $this->matrixOperator;
        return $mo->min($x,$axis);
    }

    public function argMax(NDArray $x,int $axis=null,$dtype=null)
    {
        if($axis===null) {
            return $this->la->argMax($x);
        } else {
            return $this->la->reduceArgMax($x,$axis,null,$dtype);
        }
    }

    public function argMin(NDArray $x,int $axis=null)
    {
        $mo = $this->matrixOperator;
        return $mo->argMin($x,$axis);
    }

    public function rand($shape)
    {
        $mo = $this->matrixOperator;
        return $mo->random()->rand($shape);
    }

    public function randomChoice($a,int $size=null, bool $replace=null)
    {
        $mo = $this->matrixOperator;
        return $mo->random()->choice($a, $size, $replace);
    }

    public function dot($x,$y)
    {
        return $this->la->gemm($x,$y);
    }

    public function gemm(NDArray $a,NDArray $b,float $alpha=null,float $beta=null,NDArray $c=null,$transA=null,$transB=null)
    {
        return $this->la->gemm($a, $b,$alpha,$beta,$c,$transA,$transB);
    }

    public function batch_gemm(NDArray $a,NDArray $b,float $alpha=null,float $beta=null,NDArray $c=null)
    {
        $batchSize = $a->shape()[0];
        if($c==null) {
            $outputs = null;
        } else {
            $outputs = $this->alloc(array_merge([$batchSize],$c->shape()));
            $this->la->duplicate($c,null,null,$outputs);
        }
        return $this->la->gemm($a, $b, $alpha, $beta, $outputs);
    }

    public function oneHot(NDArray $indices, int $num_classes) : NDArray
    {
        $mo = $this->matrixOperator;
        if($indices->ndim()!=1) {
            throw new InvalidArgumentException('indices must be 1-D NDarray');
        }
        $batchSize = $indices->shape()[0];
        $oneHot = $this->zeros([$batchSize,$num_classes]);
        return $mo->update($oneHot,'=',1.0,$mo->arange($batchSize),$indices);
    }

    public function relu($x) : NDArray
    {
        return $this->maximum($x,0.0);
    }

    public function sigmoid(NDArray $inputs) : NDArray
    {
        //  1 / (1.0+exp(-$x))
        $X = $this->la->copy($inputs);
        return $this->la->reciprocal(
                $this->la->exp($this->la->scal(-1.0,$X)),1.0);
    }

    public function dSigmoid(NDArray $dOutputs, NDArray $outputs) : NDArray
    {
        // dx = dy * ( 1 - y ) * y
        $this->la->multiply($dOutputs,$this->la->multiply($outputs,
            $this->la->increment($this->la->copy($outputs),1.0,-1.0)));
    }

    public function softmax(NDArray $X) : NDArray
    {
        // Yk = exp(Ak + C') / sum(exp(Ai + C'))
        $ndim = $X->ndim();

        if($ndim == 1) {
            //$X = $this->la->increment($this->la->copy($X), -$this->la->max($X)); # fix overflow
            //$expX = $this->la->exp($X);
            //return $this->la->scal(1/$this->la->sum($expX),$expX);

            // Native softmax function !!!
            return $this->la->softmax($this->la->copy($X)->reshape([1,$X->size()]))
                ->reshape([$X->size()]);
        } elseif($ndim == 2) {

            //$X = $this->la->add($this->la->reduceMax($X, $axis=1),
            //                    $this->la->copy($X),-1,$trans=true);  # fix overflow
            //$expX = $this->la->exp($X);
            //$Y = $this->la->multiply(
            //    $this->la->reciprocal($this->la->reduceSum($expX, $axis=1)),
            //    $expX,$trans=true);
            //return $Y;

            // Native softmax function !!!
            return $this->la->softmax($this->la->copy($X));

        } else {
            throw new InvalidArgumentException('Array must be 1-D or 2-D.');
        }
    }

    public function dSoftmax(NDArray $dOutputs, NDArray $outputs) : NDArray
    {
        // dx = (y * dy) - sum(y * dy) * y
        $dx = $this->la->multiply($outputs, $this->la->copy($dOutputs));
        $dInputs = $this->la->axpy(
            $this->la->multiply($this->sum($dx, $axis=1),
                                $this->la->copy($outputs),$trans=true),$dx);
        return $dInputs;
    }

    public function meanSquaredError(NDArray $trues, NDArray $predicts) : float
    {
        //  E = (1/N) * sum((Yk-Tk)**2)
        $N = $predicts->size();
        return $this->la->sum($this->la->square(
            $this->la->axpy($trues,$this->la->copy($predicts),-1.0))) / $N;
    }

    public function dMeanSquaredError(NDArray $trues,NDArray $predicts) : NDarray
    {
        // dx = 2/N * (Yk-Tk)
        return $this->la->scal(2/$predicts->size(),
            $this->la->axpy($trues,$this->la->copy($predicts),-1.0));
    }

    public function sparseCrossEntropyError(NDArray $trues, NDArray $predicts) : float
    {
        //  E = - 1/N * sum-n(sum-k(t-nk * log(y-nk)))
        $batchSize = $trues->shape()[0];
        return -1.0 * $this->la->sum($this->log($this->increment(
                $this->la->selectAxis1($predicts,$trues),
                $this->epsilon))) / $batchSize;
    }

    public function dSoftmaxSparseCrossEntropyError(NDArray $trues,NDArray $outputs) : NDArray
    {
        // dx = (y - t) / batch_size     #  t=onehot(trues) y=outputs-of-softmax
        $batchSize = $outputs->shape()[0];
        $numClass = $outputs->shape()[1];
        $dInputs = $this->la->copy($outputs);
        $this->la->onehot($trues,$numClass,-1,$dInputs);
        $dInputs = $this->scale(1/$batchSize, $dInputs);
        return $dInputs;
    }

    public function crossEntropyError(NDArray $trues, NDArray $predicts) : float
    {
        $trues = $this->la->reduceArgMax($trues,$axis=1);
        return $this->sparseCrossEntropyError($trues, $predicts);
    }

    public function dSoftmaxCrossEntropyError(NDArray $outputs,NDArray $trues) : NDArray
    {
        //  dx = (softmaxOutputs - trues) / batch_size
        $batchSize = $outputs->shape()[0];
        $dInput = $this->la->scal(1/$batchSize,
            $this->la->axpy($trues, $this->la->copy($outputs), -1));
        return $dInput;
    }

    public function equalTest($a,$b)
    {
        $mo = $this->matrixOperator;
        if($a instanceof NDArray) {
            if(!($b instanceof NDArray))
                throw new InvalidArgumentException('NDArrays must be of the same type.');
            if($a->shape()!=$b->shape())
                return false;
            $delta = $mo->zerosLike($b);
            $this->la->copy($b,$delta);
            $this->la->axpy($a,$delta,-1.0);
            $delta = $this->la->asum($delta);
        } elseif(is_numeric($a)) {
            if(!is_numeric($b))
                throw new InvalidArgumentException('Values must be of the same type.');
            $delta = abs($a - $b);
        } elseif(is_bool($a)) {
            if(!is_bool($b))
                throw new InvalidArgumentException('Values must be of the same type.');
            $delta = ($a==$b)? 0 : 1;
        } else {
            throw new InvalidArgumentException('Values must be DNArray or float or int.');
        }

        if($delta < $this->equalEpsilon) {
            return true;
        } else {
            return false;
        }
    }
}
