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

    /**
     * @param array<int> $kernelShape
     */
    abstract protected function buildNoTrainingMode(array $kernelShape) : void;

    //protected $trainingAware = true;

    protected int $axis;
    protected float $epsilon;
    protected bool $center;
    protected bool $scale;
    protected string $betaInitializerName;
    protected string $gammaInitializerName;
    protected mixed $betaInitializer;
    protected mixed $gammaInitializer;

    //protected $calcAxis;
    protected ?NDArray $beta=null;
    protected ?NDArray $gamma=null;
    protected ?NDArray $dBeta=null;
    protected ?NDArray $dGamma=null;

    //protected $xc;
    //protected $xn;
    //protected $std;
    /** @var array<int> $orignalShape1 */
    protected array $orignalShape1;
    /** @var array<int> $orignalShape2 */
    protected array $orignalShape2;
    protected int $transformShapePhase1Pre=0;
    protected int $transformShapePhase1Post=0;
    protected int $transformShapePhase2Pre=0;
    protected int $transformShapePhase2Post=0;

    public function __construct(
        object $backend,
        int $axis=null,
        float $epsilon=null,
        bool $center=null,
        bool $scale=null,
        string|callable $beta_initializer=null,
        string|callable $gamma_initializer=null,
        )
    {
        parent::__construct($backend);
        $axis = $axis ?? -1;
        $epsilon = $epsilon ?? 0.001;
        $center = $center ?? true;
        $scale = $scale ?? true;
        $beta_initializer = $beta_initializer ?? 'zeros';
        $gamma_initializer = $gamma_initializer ?? 'ones';

        $K = $this->backend;
        $this->axis = $axis;
        $this->epsilon = $epsilon;
        $this->center = $center;
        $this->scale = $scale;
        $this->betaInitializerName  = $this->toStringName($beta_initializer);
        $this->gammaInitializerName = $this->toStringName($gamma_initializer);
        $this->betaInitializer  = $K->getInitializer($beta_initializer);
        $this->gammaInitializer = $K->getInitializer($gamma_initializer);
    }

    public function build(mixed $variable=null, array $sampleWeights=null) : void
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

        //$this->calcAxis = $axis;
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

    protected function transformShape(NDArray $inputs) : NDArray
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

    protected function untransformShape(NDArray $inputs) : NDArray
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
