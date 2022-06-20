<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use ArrayAccess;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;
use Rindow\NeuralNetworks\Gradient\Core\Variable;
use Rindow\NeuralNetworks\Gradient\Core\GradientTape;
use Rindow\NeuralNetworks\Gradient\Core\GradientUtils;

class Attention extends AbstractLayerBase
{
    const RETURN_ATTENTION_SCORES = 'return_attention_scores';
    use GenericUtils;
    use GradientUtils;
    protected $backend;
    protected $scoresShape;
    //protected $returnAttentionScores;

    //protected $query;
    //protected $value;
    //protected $key;
    //protected $attentionWeight;

    public function __construct(
        object $backend,
        array $input_shapes=null,
        string $name=null,
    )
    {
        // defaults
        $input_shapes = $input_shapes ?? null;
        $name = $name ?? null;

        $this->backend = $K = $backend;
        $this->inputShape = $input_shapes;
        $this->initName($name,'attention');
    }

    public function build($variables=null, array $sampleWeights=null)
    {
        $K = $this->backend;
        $inputShapes = $this->normalizeInputShape($variables);
        if(count($inputShapes)!=2&&count($inputShapes)!=3) {
            throw new InvalidArgumentException('num of inputs must be 2 or 3: inputs is '.count($inputShapes));
        }
        foreach ($inputShapes as $idx => $shape) {
            if(!is_array($shape)||count($shape)!=2) {
                if(is_array($shape)) {
                    $type = '['.implode(',',$shape).']';
                } else {
                    $type = gettype($shape);
                }
                throw new InvalidArgumentException('input_shapes must be the list of shape: '.$type.' included in #'.$idx.'.');
            }
        }
        [$tq, $dim] = $inputShapes[0];  // Query
        [$tv, $tdim] = $inputShapes[1]; // Value
        if($dim!=$tdim) {
            throw new InvalidArgumentException('Unmatch query shape and value shape:'.
            '['.implode(',',$inputShapes[0]).'],['.implode(',',$inputShapes[1]).']');
        }
        if(count($inputShapes)==3) {
            if($inputShapes[1]!=$inputShapes[2]) {
                throw new InvalidArgumentException('value shape and key shape must be same.');
            }
        }
        $this->outputShape = [$tq,$dim];
        $this->scoresShape = [$tq,$tv];
    }

    public function getConfig() : array
    {
        return [
            'options' => [
                'input_shapes'=>$this->inputShape,
            ],
        ];
    }

    protected function assertInputShapes(array $inputs,$direction)
    {
        if(!$this->shapeInspection)
            return;
        if($this->inputShape===null) {
            throw new InvalidArgumentException('Uninitialized input shape in '.$this->name.':'.$direction);
        }
        if(count($inputs)!=2 && count($inputs)!=3) {
            throw new InvalidArgumentException('Must have 2 or 3 arguments in '.$this->name.':'.$direction);
        }
        $tq = $this->inputShape[0][0];
        $dim = $this->inputShape[0][1];
        $tv = $this->inputShape[1][0];
        $qshape = $inputs[0]->shape();
        if($qshape[1]!=$tq||$qshape[2]!=$dim){
            throw new InvalidArgumentException('Unmatch query shape [b,'.$tq.','.$dim.'] NDArray.'.
                ' ['.implode(',',$qshape).'] given in '.$this->name.':'.$direction);
        }
        if($qshape[1]!=$tq||$qshape[2]!=$dim){
            throw new InvalidArgumentException('Unmatch query shape [b,'.$tq.','.$dim.'] NDArray.'.
                ' ['.implode(',',$qshape).'] given in '.$this->name.':'.$direction);
        }
        $vshape = $inputs[0]->shape();
        if($vshape[1]!=$tq||$vshape[2]!=$dim){
            throw new InvalidArgumentException('Unmatch value shape [b,'.$tq.','.$dim.'] NDArray.'.
                ' ['.implode(',',$vshape).'] given in '.$this->name.':'.$direction);
        }
        if($qshape[0]!=$vshape[0]) {
            throw new InvalidArgumentException('Unmatch batch size.:'.
                ' ['.implode(',',$qshape).'],['.implode(',',$vshape).'] given in '.$this->name.':'.$direction);
        }
        if(count($inputs)==3) {
            $kshape = $inputs[0]->shape();
            if($kshape!=$vshape){
                throw new InvalidArgumentException('Unmatch value shape and key shape.:'.
                    ' ['.implode(',',$vshape).'],['.implode(',',$kshape).'] in '.$this->name.':'.$direction);
            }
        }
    }

    protected function assertScoresShape(NDArray $scores,$direction)
    {
        if(!$this->shapeInspection)
            return;
        if($this->scoresShape===null) {
            throw new InvalidArgumentException('Uninitialized scores shape');
        }
        $shape = $scores->shape();
        $batchNum = array_shift($shape);
        if($shape!=$this->scoresShape) {
            $shape = $this->shapeToString($shape);
            $scoresShape = $this->shapeToString($this->scoresShape);
            throw new InvalidArgumentException('unmatch scores shape: '.$shape.', must be '.scoresShape.' in '.$this->name.':'.$direction);
        }
    }

    public function backward(array $dOutputs,ArrayAccess $grads=null,array $oidsToCollect=null) : array
    {
        if(count($dOutputs)!=1) {
            throw new InvalidArgumentException('dOutputs must be list containing one NDArray');
        }
        $dOutputs = $dOutputs[0];
        if(!($dOutputs instanceof NDArray)) {
            throw new InvalidArgumentException('dOutputs must be list containing one NDArray');
        }
        $this->assertOutputShape($dOutputs,'backward');
        $dInputs = $this->differentiate($dOutputs);
        $this->assertInputShapes($dInputs,'backward');
        $this->collectGradients($this->backend,array_map(null,$this->trainableVariables(),$this->getGrads()),
            $grads,$oidsToCollect);
        return $dInputs;
    }

    protected function call(array $inputs, bool $training, array $options=null)
    {
        $K = $this->backend;
        $container = $this->container();
        $query = $inputs[0];
        $value = $inputs[1];
        if(count($inputs)==3) {
            $key = $inputs[2];
            $container->sameKey = false;
        } else {
            $key = $inputs[1];
            $container->sameKey = true;
        }
        // scores = query * key
        $scores = $K->matmul($query, $key, null, $tranB=true);
        // weights = softmax(scores)
        $attentionWeight = $K->softmax($scores);
        // vector = weights * value
        $contextVector = $K->matmul($attentionWeight, $value);

        $container->value = $value;
        $container->attentionWeight = $attentionWeight;
        $container->query = $query;
        $container->key = $key;
        if($options!==null &&
            array_key_exists(self::RETURN_ATTENTION_SCORES,$options) &&
            $options[self::RETURN_ATTENTION_SCORES]) {
            return [$contextVector,$attentionWeight];
        } else {
            return $contextVector;
        }
    }

    protected function differentiate(NDArray $dOutputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        // forward:
        //   vector = weights (*) value
        // backward:
        //   dWeights = dVector (*) value^T
        //   dValue   = weights^T (*) dVector
        $dAttentionWeight = $K->matmul($dOutputs,$container->value,$transA=false,$transB=true);
        $dValue = $K->matmul($container->attentionWeight,$dOutputs,$transA=true,$transB=false);
        $dScores = $K->dSoftmax($dAttentionWeight,$container->attentionWeight);

        $dQuery = $K->matmul($dScores,$container->key,$transA=false,$transB=false);
        $dKey = $K->matmul($dScores,$container->query,$transA=true,$transB=false);

        if($container->sameKey) {
            $K->update_add($dValue,$dKey);
            return [$dQuery,$dValue];
        } else {
            return [$dQuery,$dValue,$dKey];
        }
    }

    protected function numOfOutputs($options)
    {
        if($options[self::RETURN_ATTENTION_SCORES] ?? false) {
            return 2;
        }
        return 1;
    }

    public function __invoke($inputs, $training, $options=null)
    {
        $outputs = $this->forward($inputs, $training, $options);
        return $outputs;
    }

    /**
    *  @param array<Variable>  $inputs
    *  @return array<Variable>|Variable
    */
    public function forward(array $inputs, Variable|bool $training, array $options=null)
    {
        //$outputs = null;
        if(!is_array($inputs)) {
            throw new InvalidArgumentException('inputs must be list of Variable');
        }
        [$inputs,$rawInputs]     = $this->packAndUnpackVariables($this->backend,$inputs);
        [$training,$rawTraining] = $this->packAndUnpackVariable($this->backend,$training);
        if(!$this->built) {
            $this->build($inputs);
            $this->built = true;
        }
        $numOfOutputs = $this->numOfOutputs($options);
        $session = $this->preGradientProcessOnSession($inputs);
        $session->begin();
        try {
            $this->assertInputShapes($rawInputs,'forward');
            $rawOutputs = $this->call($rawInputs,$rawTraining,$options);
            if($options!==null &&
                array_key_exists(self::RETURN_ATTENTION_SCORES,$options)&&
                $options[self::RETURN_ATTENTION_SCORES]) {
                $this->assertOutputShape($rawOutputs[0],'forward');
                $this->assertScoresShape($rawOutputs[1],'forward');
            } else {
                $this->assertOutputShape($rawOutputs,'forward');
            }
        } finally{
            $session->end();
        }
        if($numOfOutputs>1) {
            [$rawOutputs,$scores] = $rawOutputs;
        }
        $outputs = $this->postGradientProcessOnSession(
            $this->backend, $session,$inputs, [$rawOutputs]);
        if($numOfOutputs>1) {
            array_push($outputs,$scores);
            return $outputs;
        } else {
            return $outputs[0];
        }
    }

    /**
     * Call from SessionFunc in compiled graph
     */
    public function _rawCall(array $inputs,array $options)
    {
        $training = $options['training'] ?? false;
        $opts=[];
        $opts[self::RETURN_ATTENTION_SCORES] = $options[self::RETURN_ATTENTION_SCORES] ?? false;
        $outputs = $this->call($inputs, $training, $opts);
        if(!is_array($outputs)) {
            $outputs = [$outputs];
        }
        return $outputs;
    }

}
