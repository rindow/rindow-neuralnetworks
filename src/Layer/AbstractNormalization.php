<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

abstract class AbstractNormalization extends AbstractLayer
{
    use GenericUtils;

    abstract protected function call(NDArray $inputs, ?bool $training=null) : NDArray;

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
    protected ?array $orignalShape1;
    /** @var array<int> $orignalShape2 */
    protected ?array $orignalShape2;
    protected int $transformShapePhase1Pre=0;
    protected int $transformShapePhase1Post=0;
    protected int $transformShapePhase2Pre=0;
    protected int $transformShapePhase2Post=0;

    public function __construct(
        object $backend,
        ?int $axis=null,
        ?float $epsilon=null,
        ?bool $center=null,
        ?bool $scale=null,
        string|callable|null $beta_initializer=null,
        string|callable|null $gamma_initializer=null,
        )
    {
        parent::__construct($backend);
        $axis = $axis ?? -1;
        $epsilon ??= 1e-3;
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

    public function build(mixed $variable=null, ?array $sampleWeights=null) : void
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
        //echo "ndim={$ndim}\n";
        //echo "axis={$axis}\n";
        //echo "inputShape=".$this->shapeToString($inputShape)."\n";
        //if($ndim>$axis) {
        //    $nnn = array_slice($inputShape,0,$axis);
        //    $this->transformShapePhase1Pre = (int)array_product($nnn);
        //    $this->transformShapePhase1Post = (int)array_product(array_slice($inputShape,$axis));
        //    echo "Phase1Pre={$this->transformShapePhase1Pre}\n";
        //    echo "Phase1Post={$this->transformShapePhase1Post}\n";
        //}
        //if($ndim>1) {
        //    $this->transformShapePhase2Pre = (int)(array_product($inputShape)/$featureSize);
        //    $this->transformShapePhase2Post = $featureSize;
        //    echo "Phase2Pre={$this->transformShapePhase2Pre}\n";
        //    echo "Phase2Post={$this->transformShapePhase2Post}\n";
        //}
        $this->inputShape = $inputShape;
        $this->outputShape = $inputShape;
        $this->syncWeightVariables();
    }

    protected function transformShape(NDArray $inputs) : NDArray
    {
        $K = $this->backend;
        $orignalShape = $inputs->shape();

        $axis = $this->axis;
        //echo "@axis={$axis}\n";
        $ndim = $inputs->ndim();
        if($axis<0) {
            $axis = $ndim+$axis;
        }
        if($axis<1) {
            throw new InvalidArgumentException('Axis must be greater than 0');
        }
        $full_input_shape = $inputs->shape();
        //echo "ndim={$ndim}\n";
        //echo "axis={$axis}\n";
        //echo "full_input_shape=".$this->shapeToString($full_input_shape)."\n";
        $transformShapePhase1Pre=0;
        $transformShapePhase1Post=0;
        $transformShapePhase2Pre=0;
        $transformShapePhase2Post=0;
            
        if($ndim-1>$axis) {
            //echo "try ph1\n";
            $outerShape = $full_input_shape;
            $innerShape = array_splice($outerShape,$axis+1);
            //echo "innerShape=".$this->shapeToString($innerShape)."\n";
            //echo "outerShape=".$this->shapeToString($outerShape)."\n";
            $transformShapePhase1Pre = (int)array_product($outerShape);
            $transformShapePhase1Post = (int)array_product($innerShape);
            //echo "Phase1Pre={$transformShapePhase1Pre}\n";
            //echo "Phase1Post={$transformShapePhase1Post}\n";
        }
        if($ndim>2) {
            //echo "try ph2\n";
            $outerShape = $full_input_shape;
            $innerShape = array_splice($outerShape,$axis,1);
            //echo "innerShape=".$this->shapeToString($innerShape)."\n";
            //echo "outerShape=".$this->shapeToString($outerShape)."\n";
            $transformShapePhase2Pre = (int)array_product($outerShape);
            $transformShapePhase2Post = (int)array_product($innerShape);
            //echo "Phase2Pre={$transformShapePhase2Pre}\n";
            //echo "Phase2Post={$transformShapePhase2Post}\n";
        }

        $this->orignalShape1 = null;
        if($transformShapePhase1Pre) {
            $this->orignalShape1 = $inputs->shape();
            //echo "transpose:".$this->shapeToString($inputs->shape());
            $inputs = $inputs->reshape([
                $transformShapePhase1Pre,
                $transformShapePhase1Post,
            ]);
            //echo "->".$this->shapeToString($inputs->shape());
            $inputs = $K->transpose($inputs);
            //echo "->".$this->shapeToString($inputs->shape())."\n";
            //echo "pre transpose\n";
        }
        $this->orignalShape2 = null;
        if($transformShapePhase2Pre) {
            $this->orignalShape2 = $inputs->shape();
            $inputs = $inputs->reshape([
                $transformShapePhase2Pre,
                $transformShapePhase2Post,
            ]);
            //echo "pre reshape\n";
        }
        //echo "transform=".$this->shapeToString($inputs->shape())."\n";
        if($inputs->ndim()!=2) {
            throw new InvalidArgumentException("Invalid shape of inputs: ".
                $this->shapeToString($orignalShape)." given and translated to ".$this->shapeToString($inputs->shape()));
        }
        return $inputs;
    }

    protected function untransformShape(NDArray $inputs) : NDArray
    {
        $K = $this->backend;
        if($this->orignalShape2) {
            $inputs = $inputs->reshape($this->orignalShape2);
            //echo "post reshape\n";
        }
        if($this->orignalShape1) {
            $inputs = $K->transpose($inputs);
            $inputs = $inputs->reshape($this->orignalShape1);
            //echo "post transpose\n";
        }
        //echo "untransform=".$this->shapeToString($inputs->shape())."\n";
        return $inputs;
    }

}
