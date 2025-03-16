<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class Concatenate extends AbstractMultiInputLayer
{
    use GenericUtils;

    protected int $axis;
    /** @var array<int|array<int>> $shapes */
    protected $shapes;

    /**
     * @param array<array<int>> $input_shapes
     */
    public function __construct(
        object $backend,
        ?int $axis=null,
        ?array $input_shapes=null,
        ?string $name=null,
    )
    {
        // defaults
        $axis = $axis ?? -1;
        $input_shapes = $input_shapes ?? null;
        $name = $name ?? null;

        parent::__construct($backend);
        $this->axis = $axis;
        $this->inputShape = $input_shapes;
        $this->initName($name,'concatenate');
    }

    public function build(mixed $variables=null, ?array $sampleWeights=null) : void
    {
        $K = $this->backend;
        if(!is_array($variables) && $variables!==null) {
            throw new InvalidArgumentException('inputs must be list of variable');
        }
        $inputShapes = $this->normalizeInputShapes($variables);
        if(count($inputShapes)<2) {
            throw new InvalidArgumentException('num of inputs must be greater then 2 or equal: input dims is '.count($inputShapes));
        }
        $m = 0;
        $baseShape = null;
        foreach ($inputShapes as $idx => $shape) {
            if(!is_array($shape)) {
                throw new InvalidArgumentException('input_shapes must be the list of shape:');
            }
            if($this->axis < 0) {
                $axis = 1 + count($shape) + $this->axis;
            } else {
                $axis = $this->axis;
            }
            $shapestack = [];
            // skip batchsize and axis
            for($i=1;$i<$axis;$i++) {
                $shapestack[] = array_shift($shape);
            }
            $m += array_shift($shape);
            if($baseShape===null) {
                $baseShape = array_merge($shapestack,$shape);
            } else {
                if($baseShape != array_merge($shapestack,$shape)) {
                    $msg = '';
                    foreach($inputShapes as $shape) {
                        $msg .= '['.implode(',',$shape).'] ';
                    }
                    throw new InvalidArgumentException('Unmatch input_shapes:'.$msg);
                }
            }
        }
        array_unshift($shape,$m);
        $shape = array_merge($shapestack,$shape);
        $this->outputShape = $shape;
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
            'options' => [
                'axis'=>$this->axis,
                'input_shapes'=>$this->inputShape,
            ]
        ];
    }

    protected function call(array $inputs, ?bool $training=null) : NDArray
    {
        $K = $this->backend;
        $container = $this->container();
        $outputs = $K->concat($inputs,$this->axis);
        $container->shapes = [];
        foreach ($inputs as $v) {
            $container->shapes[] = $v->shape();
        }
        return $outputs;
    }

    protected function differentiate(NDArray $dOutputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        if($this->axis<0) {
            $axis = count($container->shapes[0])+$this->axis;
        } else {
            $axis = $this->axis;
        }
        $sizeSplits = [];
        foreach ($container->shapes as $shape) {
            $sizeSplits[] = $shape[$axis];
        }
        $dInputs = $K->split($dOutputs,$sizeSplits,$axis);
        return $dInputs;
    }
}
