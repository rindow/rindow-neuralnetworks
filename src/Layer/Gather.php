<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class Gather extends AbstractMultiInputLayer
{
    use GenericUtils;
    protected ?int $axis;
    protected ?int $batchDims;
    protected ?int $detailDepth;
    protected ?int $indexDepth;
    protected int $realAxis;
    protected int $realBatchDims;
    protected int $realDetailDepth;
    protected int $realIndexDepth;
    protected int $reduceNumClass;

    /**
     * @param array<array<int>> $input_shapes
     */
    public function __construct(
        object $backend,
        ?int $axis=null,
        ?int $batchDims=null,
        ?int $detailDepth=null,
        ?int $indexDepth=null,
        ?array $input_shapes=null,
        ?string $name=null,
    )
    {
        // defaults
        $input_shapes = $input_shapes ?? null;
        $name = $name ?? null;
        
        parent::__construct($backend);
        $this->axis = $axis;
        $this->batchDims = $batchDims;
        $this->detailDepth = $detailDepth;
        $this->indexDepth = $indexDepth;
        $this->inputShape = $input_shapes;
        $this->initName($name,'gather');
    }

    public function build(mixed $variables=null, ?array $sampleWeights=null) : void
    {
        $K = $this->backend;

        $inputShapes = $this->normalizeInputShapes($variables);
        if(count($inputShapes)!=2) {
            throw new InvalidArgumentException('num of inputs must be 2: inputs is '.count($inputShapes));
        }
        foreach ($inputShapes as $idx => $shape) {
            if(!is_array($shape)) {
                $type = gettype($shape);
                throw new InvalidArgumentException('input_shapes must be the list of shape: '.$type.' included in #'.$idx.'.');
            }
        }
        [$paramShape,$indexShape] = $inputShapes;
        //if($this->axis===null) {
        //    throw new InvalidArgumentException('Null axis is not supported.');
        //}
        //$mo = $this->backend->localMatrixOperator();
        //echo "paramShape:".$mo->shapeToString($paramShape)."\n";
        //echo "indexShape:".$mo->shapeToString($indexShape)."\n";

        $axis = $this->axis;
        $batchDims = $this->batchDims;
        $detailDepth = $this->detailDepth;
        $indexDepth = $this->indexDepth;

        //
        // Calculate Dims Defaults
        //
        //if($axis < 0) {
        //    $axis = count($paramShape) + $axis;
        //}
        //if($axis<0||$axis>count($paramShape)) {
        //    throw new InvalidArgumentException(
        //        'Invalid axis. Dims of the paramShape is '.count($paramShape).'. axis='.$this->axis.' given');
        //}
        $batchDims ??= 0;
        if($batchDims<0) {
            $batchDims += count($paramShape);
        }
        if($batchDims<0 || $batchDims>=count($paramShape)) {
            throw new InvalidArgumentException(
                "batchDims($batchDims) must be less than to ndims of params (".count($paramShape).") in input_shapes"
            );
        }
        $axis ??= $batchDims;
        if($axis<0) {
            $axis += count($paramShape);
        }
        if($axis<0 || $axis>=count($paramShape)) {
            throw new InvalidArgumentException(
                "axis ($axis) must be less than to ndims of params (".count($paramShape).")"
            );
        }
        $detailDepth ??= $axis+1;
        if($detailDepth<0) {
            $detailDepth += count($paramShape);
        }
        if($detailDepth<0 || $detailDepth>count($paramShape)) {
            throw new InvalidArgumentException(
                "axis ($detailDepth) must be less than or equal to ndims of params (".count($paramShape).")"
            );
        }
        $indexDepth ??= count($indexShape);
        if($indexDepth<0) {
            $indexDepth += count($indexShape);
        }
        if($indexDepth<0 || $indexDepth>count($indexShape)) {
            throw new InvalidArgumentException(
                "axis ($indexDepth) must be less than or equal to ndims of params (".count($indexShape).")"
            );
        }
        if($batchDims>$axis) {
            if($axis==0) {
                $batchDims = 0;
            } else {
                throw new InvalidArgumentException("batchDims ($batchDims) must be less than or equal to axis ($axis)");
            }
        }

        // 
        // Parsing shapes
        //
        // paramShape
        $batchShape = $paramShape;
        $outerShape = array_splice($batchShape, $batchDims);
        $innerShape = array_splice($outerShape, $axis-$batchDims);
        $numClass = array_shift($innerShape);
        $detailShape = array_splice($innerShape, $detailDepth-1-$axis);
        // indexShape
        $batchShapeX = $indexShape;
        $indexShape = array_splice($batchShapeX, $batchDims);
        $innerShapeX = array_splice($indexShape, $indexDepth-$batchDims);
        if($batchShape!=$batchShapeX) {
            throw new InvalidArgumentException(
                "Unmatch batch shape of params and indices: ".
                $this->shapetoString($batchShape).",".
                $this->shapetoString($batchShapeX).","
            );
        }
        if($innerShape!=$innerShapeX) {
            throw new InvalidArgumentException(
                "Unmatch inner shape of params and indices: ".
                "param's inner shape is ".$this->shapetoString($innerShape).",".
                "index's inner shape is ".$this->shapetoString($innerShapeX).","
            );
        }
        //$batches = array_product($batchShape);
        //$m = array_product($outerShape);
        //$n = array_product($indexShape);
        //$k = array_product($innerShape);
        //$len = array_product($detailShape);
        // outputsShape
        $outputShape = array_merge($batchShape, $outerShape, $indexShape, $innerShape, $detailShape);

        //$postfixShape = $paramShape;
        //$prefixShape = [];
        //for($i=0;$i<$axis;$i++) {
        //    $prefixShape[] = array_shift($postfixShape);
        //}
        //$this->reduceNumClass = array_shift($postfixShape);
        //$outputShape = array_merge($prefixShape,$postfixShape);
        //if($indexShape!=$outputShape) {
        //    throw new InvalidArgumentException('Unmatch params and index Shape and axis:'.
        //            $this->shapeToString($paramShape).','.
        //            $this->shapeToString($outputShape).','.$this->axis);
        //}

        $this->realAxis = $axis+1;
        $this->realBatchDims = $batchDims+1;
        $this->realDetailDepth = $detailDepth+1;
        $this->realIndexDepth = $indexDepth+1;

        $this->outputShape = $outputShape;
    }

    public function getParams() : array
    {
        return [];
    }

    public function getGrads() : array
    {
        return [];
    }

    public function getConfig() : array
    {
        return [
            'axis' => $this->axis,
            'options' => [
                'input_shape'=>$this->inputShape,
            ]
        ];
    }

    protected function call(array $inputs, ?bool $training=null) : NDArray
    {
        $K = $this->backend;
        $container = $this->container();
        [$params,$indices] = $inputs;
        //$outputs = $K->gather($params,$indices,$this->realAxis);
        $outputs = $K->gatherb(
            $params,
            $indices,
            axis:$this->realAxis,
            batchDims:$this->realBatchDims,
            detailDepth:$this->realDetailDepth,
            indexDepth:$this->realIndexDepth,
        );
        $container->indices = $indices;
        $container->orignalParamShape = $params->shape();
        return $outputs;
    }

    protected function differentiate(NDArray $dOutputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        //$dParams = $K->scatter(
        //    $container->indices,
        //    $dOutputs,
        //    $this->reduceNumClass,
        //    $this->realAxis
        //);
        $dParams = $K->scatterb(
            $container->indices,
            $dOutputs,
            $container->orignalParamShape,
            axis:$this->realAxis,
            batchDims:$this->realBatchDims,
            detailDepth:$this->realDetailDepth,
            indexDepth:$this->realIndexDepth,
        );
        $dIndices = $K->zerosLike($container->indices);
        return [$dParams,$dIndices];
    }

}
