<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class BatchNormalization extends AbstractLayer implements Layer
{
    use GenericUtils;
    protected $backend;
    protected $axis;
    protected $momentum;
    protected $epsilon;
    protected $center;
    protected $scale;
    protected $betaInitializerName;
    protected $gammaInitializerName;
    protected $movingMeanInitializerName;
    protected $movingVarianceInitializerName;
    protected $betaInitializer;
    protected $gammaInitializer;
    protected $movingMeanInitializer;
    protected $movingVarianceInitializer;

    protected $calcAxis;
    protected $beta;
    protected $gamma;
    protected $dBeta;
    protected $dGamma;
    protected $movingMean;
    protected $xc;
    protected $xn;
    protected $std;
    protected $transformShapePhase1Pre;
    protected $transformShapePhase1Post;
    protected $transformShapePhase2Pre;
    protected $transformShapePhase2Post;

    public function __construct($backend,array $options=null)
    {
        extract($this->extractArgs([
            'axis'=>-1,
            'momentum'=>0.99,
            'epsilon'=>0.001,
            'center'=>true,
            'scale'=>true,
            'beta_initializer'=>'zeros',
            'gamma_initializer'=>'ones',
            'moving_mean_initializer'=>'zeros',
            'moving_variance_initializer'=>'ones',
        ],$options));
        $this->backend = $K = $backend;
        $this->axis = $axis;
        $this->momentum = $momentum;
        $this->epsilon = $epsilon;
        $this->center = $center;
        $this->scale = $scale;
        $this->betaInitializerName  = $beta_initializer;
        $this->gammaInitializerName = $gamma_initializer;
        $this->movingMeanInitializerName  = $moving_mean_initializer;
        $this->movingVarianceInitializerName = $moving_variance_initializer;
        $this->betaInitializer  = $K->getInitializer($beta_initializer);
        $this->gammaInitializer = $K->getInitializer($gamma_initializer);
        $this->movingMeanInitializer  = $K->getInitializer($moving_mean_initializer);
        $this->movingVarianceInitializer = $K->getInitializer($moving_variance_initializer);
    }

    public function build($variable=null, array $options=null)
    {
        extract($this->extractArgs([
            'sampleWeights'=>null,
        ],$options));
        $K = $this->backend;
        $betaInitializer = $this->betaInitializer;
        $gammaInitializer = $this->gammaInitializer;
        $movingMeanInitializer = $this->movingMeanInitializer;
        $movingVarianceInitializer = $this->movingVarianceInitializer;

        // ********* CAUTION ***********
        //  if inappropriate, then Return old varsion shape normarization
        $inputShape = $this->normalizeInputShape($variable);
        $axis = $this->axis;
        $ndim = count($inputShape);
        if($axis<0) {
            $axis = $ndim+1+$axis;
        }
        if($axis<1) {
            throw new InvalidArgumentException('Axis must be greater than 0');
        }
        $featureSize = $inputShape[$axis-1];
        $kernelShape = [$featureSize];
        if($sampleWeights) {
            if($this->center) {
                $this->beta = $sampleWeights[0];
            }
            if($this->scale) {
                $this->gamma = $sampleWeights[1];
            }
        } else {
            if($this->center) {
                $this->beta  = $betaInitializer($kernelShape);
            }
            if($this->scale) {
                $this->gamma = $gammaInitializer($kernelShape);
            }
        }
        if($this->center) {
            $this->dBeta = $K->zerosLike($this->beta);
        }
        if($this->scale) {
            $this->dGamma = $K->zerosLike($this->gamma);
        }

        $this->movingMean = $movingMeanInitializer($kernelShape);
        $this->movingVariance = $movingVarianceInitializer($kernelShape);
        $this->calcAxis = $axis;
        if($ndim>$axis) {
            $nnn = array_slice($inputShape,0,$axis);
            $this->transformShapePhase1Pre = (int)array_product($nnn);
            $this->transformShapePhase1Post = (int)array_product(array_slice($inputShape,$axis));
        }
        if($ndim>1) {
            $this->transformShapePhase2Pre = (int)(array_product($inputShape)/$featureSize);
            $this->transformShapePhase2Post = $featureSize;
        }
        $this->inputShape = $inputShape;
        $this->outputShape = $inputShape;
        return $this->createOutputDefinition([$this->outputShape]);
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
                'axis'=>$this->axis,
                'momentum'=>$this->momentum,
                'epsilon'=>$this->epsilon,
                'beta_initializer'=>$this->betaInitializerName,
                'gamma_initializer'=>$this->gammaInitializerName,
                'moving_mean_initializer'=>$this->movingMeanInitializerName,
                'moving_variance_initializer'=>$this->movingVarianceInitializerName,
            ]
        ];
    }

    protected function transformShape($inputs)
    {
        $K = $this->backend;
        $batches = $inputs->shape()[0];
        if($this->transformShapePhase1Pre) {
            $this->orignalShape1 = $inputs->shape();
            $inputs = $inputs->reshape([
                $batches*$this->transformShapePhase1Pre,
                $this->transformShapePhase1Post,
            ]);
            $inputs = $K->transpose($inputs);
        }
        if($this->transformShapePhase2Pre) {
            $this->orignalShape2 = $inputs->shape();
            $inputs = $inputs->reshape([
                $batches*$this->transformShapePhase2Pre,
                $this->transformShapePhase2Post,
            ]);
        }
        return $inputs;
    }

    protected function untransformShape($inputs)
    {
        $K = $this->backend;
        if($this->transformShapePhase2Pre) {
            $inputs = $inputs->reshape($this->orignalShape2);
        }
        if($this->transformShapePhase1Pre) {
            $inputs = $K->transpose($inputs);
            $inputs = $inputs->reshape($this->orignalShape1);
        }
        return $inputs;
    }

    protected function call(NDArray $inputs, bool $training) : NDArray
    {
        $K = $this->backend;
        $inputs = $this->transformShape($inputs);

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

        if($this->gamma) {
            $outputs = $K->mul($this->gamma, $xn);
        } else {
            $outputs = $xn;
        }
        if($this->beta) {
            $outputs = $K->add($outputs, $this->beta);
        }

        $outputs = $this->untransformShape($outputs);
        return $outputs;
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        $dOutputs = $this->transformShape($dOutputs);
        $numItems = $dOutputs->shape()[0];

        if($this->dBeta) {
            $dbeta = $K->sum($dOutputs,$axis=0,$this->dBeta);
            //$K->copy($dbeta,$this->dBeta);
        }
        if($this->dGamma) {
            $dgamma = $K->sum($K->mul($this->xn, $dOutputs), $axis=0,$this->dGamma);
            //$K->copy($dgamma,$this->dGamma);
            $dxn = $K->mul($this->gamma, $dOutputs);
        } else {
            $dxn = $dOutputs;
        }
        if($this->std===null)
            throw new LogicException('not initialized for training');
        $dxc = $K->div($dxn, $this->std);
        $dstd = $K->scale(-1.0, $K->sum(
            $K->div($K->mul($dxn, $this->xc), $K->mul($this->std, $this->std)),
            $axis=0));
        $dvar = $K->div($K->scale(0.5, $dstd), $this->std);
        $K->update_add($dxc,
            $K->scale(2.0/$numItems, $K->mul($this->xc, $dvar)));
        $dmu = $K->sum($dxc, $axis=0);
        $dInputs = $K->sub($dxc, $K->scale(1/$numItems,$dmu));

        $dInputs = $this->untransformShape($dInputs);
        return $dInputs;
    }
}
