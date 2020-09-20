<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class BatchNormalization extends AbstractLayer implements Layer
{
    use GenericUtils;
    protected $backend;
    protected $momentum;
    protected $epsilon;
    protected $betaInitializer;
    protected $gammaInitializer;
    protected $movingMeanInitializer;
    protected $movingVarianceInitializer;

    protected $beta;
    protected $gamma;
    protected $dBeta;
    protected $dGamma;
    protected $movingMean;
    protected $xc;
    protected $xn;
    protected $std;

    public function __construct($backend,array $options=null)
    {
        extract($this->extractArgs([
            'momentum'=>0.99,
            'epsilon'=>0.001,
            'beta_initializer'=>'zeros',
            'gamma_initializer'=>'ones',
            'moving_mean_initializer'=>'zeros',
            'moving_variance_initializer'=>'ones',
        ],$options));
        $this->backend = $K = $backend;
        $this->momentum = $momentum;
        $this->epsilon = $epsilon;
        $this->betaInitializer  = $K->getInitializer($beta_initializer);
        $this->gammaInitializer = $K->getInitializer($gamma_initializer);
        $this->movingMeanInitializer  = $K->getInitializer($moving_mean_initializer);
        $this->movingVarianceInitializer = $K->getInitializer($moving_variance_initializer);
    }

    public function build(array $inputShape=null, array $options=null) : array
    {
        extract($this->extractArgs([
            'sampleWeights'=>null,
        ],$options));
        $K = $this->backend;
        $betaInitializer = $this->betaInitializer;
        $gammaInitializer = $this->gammaInitializer;
        $movingMeanInitializer = $this->movingMeanInitializer;
        $movingVarianceInitializer = $this->movingVarianceInitializer;

        if($inputShape===null)
            $inputShape = $this->inputShape;
        if($this->inputShape===null)
            $this->inputShape = $inputShape;
        if($this->inputShape!==$inputShape) {
            throw new InvalidArgumentException(
                'Input shape is inconsistent: ['.implode(',',$this->inputShape).
                '] and ['.implode(',',$inputShape).']');
        } elseif($inputShape===null) {
            throw new InvalidArgumentException('Input shape is not defined');
        }
        if(count($inputShape)!=1) {
            throw new InvalidArgumentException(
                'Unsuppored input shape: ['.implode(',',$inputShape).']');
        }
        if($sampleWeights) {
            $this->beta = $sampleWeights[0];
            $this->gamma = $sampleWeights[1];
        } else {
            $this->beta  = $betaInitializer($inputShape);
            $this->gamma = $gammaInitializer($inputShape);
        }
        $this->dBeta = $K->zerosLike($this->beta);
        $this->dGamma = $K->zerosLike($this->gamma);

        $this->movingMean = $movingMeanInitializer($inputShape);
        $this->movingVariance = $movingVarianceInitializer($inputShape);

        $this->inputShape = $inputShape;
        $this->outputShape = $inputShape;
        return $this->outputShape;
    }

    public function getParams() : array
    {
        return [$this->beta,$this->gamma];
    }

    public function getGrads() : array
    {
        return [$this->dBeta,$this->dGamma];
    }

    public function getConfig() : array
    {
        return [
            'options' => [
                'momentum'=>0.99,
                'epsilon'=>0.001,
                'beta_initializer'=>'zeros',
                'gamma_initializer'=>'ones',
                'moving_mean_initializer'=>'zeros',
                'moving_variance_initializer'=>'ones',
            ]
        ];
    }

    protected function call(NDArray $inputs, bool $training) : NDArray
    {
        $K = $this->backend;

        // normalization
        if($training) {
            $mu = $K->mean($inputs,$axis=0);
            $xc = $K->sub($inputs, $mu);
            $v = $K->mean($K->square($xc), $axis=0);
            $std = $K->sqrt($K->increment($v, $this->epsilon));
            $xn = $K->div($xc, $std);

            $this->xc = $xc;
            $this->xn = $xn;
            $this->std = $std;
            $this->movingMean =     $K->add($K->scale($this->momentum,   $this->movingMean),
                                            $K->scale(1-$this->momentum, $mu));
            $this->movingVariance = $K->add($K->scale($this->momentum,   $this->movingVariance),
                                            $K->scale(1-$this->momentum, $v));
        } else {
            $xc = $K->sub($inputs, $this->movingMean);
            $xn = $K->div($xc, ($K->sqrt($K->increment($this->movingVariance, $this->epsilon))));
        }

        $outputs = $K->add($K->mul($this->gamma, $xn), $this->beta);

        return $outputs;
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        $batchSize = $dOutputs->shape()[0];

        $dbeta = $K->sum($dOutputs,$axis=0);
        $dgamma = $K->sum($K->mul($this->xn, $dOutputs), $axis=0);
        $dxn = $K->mul($this->gamma, $dOutputs);
        if($this->std===null)
            throw new LogicException('not initialized for training');
        $dxc = $K->div($dxn, $this->std);
        $dstd = $K->scale(-1.0, $K->sum(
            $K->div($K->mul($dxn, $this->xc), $K->mul($this->std, $this->std)),
            $axis=0));
        $dvar = $K->div($K->scale(0.5, $dstd), $this->std);
        $K->update_add($dxc,
            $K->scale(2.0/$batchSize, $K->mul($this->xc, $dvar)));
        $dmu = $K->sum($dxc, $axis=0);
        $dInputs = $K->sub($dxc, $K->scale(1/$batchSize,$dmu));

        $this->dBeta = $dbeta;
        $this->dGamma = $dgamma;

        return $dInputs;
    }
}
