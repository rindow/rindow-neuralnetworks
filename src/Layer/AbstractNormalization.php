<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

abstract class AbstractNormalization extends AbstractLayer
{
    use GenericUtils;

    abstract protected function call(NDArray $inputs, bool $training=null) : NDArray;
    abstract protected function differentiate(NDArray $dOutputs) : NDArray;
    abstract protected function buildNoTrainingMode(array $kernelShape) : void;

    protected $trainingAware = true;

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
    protected $movingVariance;
    //protected $xc;
    //protected $xn;
    //protected $std;
    protected $orignalShape1;
    protected $orignalShape2;
    protected $transformShapePhase1Pre;
    protected $transformShapePhase1Post;
    protected $transformShapePhase2Pre;
    protected $transformShapePhase2Post;

    public function build($variable=null, array $sampleWeights=null)
    {
        $K = $this->backend;
        $betaInitializer = $this->betaInitializer;
        $gammaInitializer = $this->gammaInitializer;

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
        if($this->beta===null) {
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
        }

        $this->buildNoTrainingMode($kernelShape);

        if($this->center) {
            $this->dBeta = $K->zerosLike($this->beta);
        }
        if($this->scale) {
            $this->dGamma = $K->zerosLike($this->gamma);
        }

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
        $this->syncWeightVariables();
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

}
