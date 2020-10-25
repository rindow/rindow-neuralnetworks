<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class Attention extends AbstractLayerBase
{
    use GenericUtils;
    protected $backend;
    protected $returnAttentionScores;
    protected $query;
    protected $value;
    protected $key;
    protected $scores;
    protected $attentionWeight;
    protected $scoresShape;

    public function __construct($backend, array $options=null)
    {
        extract($this->extractArgs([
            'input_shapes'=>null,
            'return_attention_scores'=>false,
        ],$options));
        $this->backend = $K = $backend;
        $this->inputShape = $input_shapes;
        $this->returnAttentionScores = $return_attention_scores;
    }

    public function build(array $inputShape=null, array $options=null) : array
    {
        $K = $this->backend;
        $inputShape = $this->normalizeInputShape($inputShape);
        if(count($inputShape)!=2&&count($inputShape)!=3) {
            throw new InvalidArgumentException('num of inputs must be 2 or 3: inputs is '.count($inputShape));
        }
        foreach ($inputShape as $shape) {
            if(!is_array($shape)||count($shape)!=2) {
                throw new InvalidArgumentException('input_shapes must be the list of shape:');
            }
        }
        [$tq, $dim] = $inputShape[0];  // Query
        [$tv, $tdim] = $inputShape[1]; // Value
        if($dim!=$tdim) {
            throw new InvalidArgumentException('Unmatch query shape and value shape:'.
            '['.implode(',',$inputShape[0]).'],['.implode(',',$inputShape[1]).']');
        }
        if(count($inputShape)==3) {
            if($inputShape[1]!=$inputShape[2]) {
                throw new InvalidArgumentException('value shape and key shape must be same.');
            }
        }
        $this->outputShape = [$tq,$dim];
        $this->scoresShape = [$tq,$tv];
        if($this->returnAttentionScores) {
            return [$this->outputShape,$this->scoresShape];
        } else {
            return $this->outputShape;
        }
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

    public function forward(array $inputs, bool $training)
    {
        $this->assertInputShapes($inputs,'forward');
        $outputs = $this->call($inputs,$training);
        if($this->returnAttentionScores) {
            $this->assertOutputShape($outputs[0],'forward');
            $this->assertScoresShape($outputs[1],'forward');
        } else {
            $this->assertOutputShape($outputs,'forward');
        }
        return $outputs;
    }

    public function backward(NDArray $dOutputs) : array
    {
        $this->assertOutputShape($dOutputs,'backward');
        $dInputs = $this->differentiate($dOutputs);
        $this->assertInputShapes($dInputs,'backward');
        return $dInputs;
    }

    protected function call(array $inputs, bool $training)
    {
        $K = $this->backend;
        $query = $inputs[0];
        $value = $inputs[1];
        if(count($inputs)==3) {
            $key = $inputs[2];
            $this->sameKey = false;
        } else {
            $key = $inputs[1];
            $this->sameKey = true;
        }
        // scores = query * key
        $scores = $K->matmul($query, $key, null, $tranB=true);
        // weights = softmax(scores)
        $attentionWeight = $K->softmax($scores);
        // vector = weights * value
        $contextVector = $K->matmul($attentionWeight, $value);

        $this->value = $value;
        $this->attentionWeight = $attentionWeight;
        $this->query = $query;
        $this->key = $key;
        if($this->returnAttentionScores) {
            return [$contextVector,$attentionWeight];
        } else {
            return $contextVector;
        }
    }

    protected function differentiate(NDArray $dOutputs) : array
    {
        $K = $this->backend;
        // forward:
        //   vector = weights (*) value
        // backward:
        //   dWeights = dVector (*) value^T
        //   dValue   = weights^T (*) dVector
        $dAttentionWeight = $K->matmul($dOutputs,$this->value,$transA=false,$transB=true);
        $dValue = $K->matmul($this->attentionWeight,$dOutputs,$transA=true,$transB=false);
        $dScores = $K->dSoftmax($dAttentionWeight,$this->attentionWeight);

        $dQuery = $K->matmul($dScores,$this->key,$transA=false,$transB=false);
        $dKey = $K->matmul($dScores,$this->query,$transA=true,$transB=false);

        if($this->sameKey) {
            $K->update_add($dValue,$dKey);
            return [$dQuery,$dValue];
        } else {
            return [$dQuery,$dValue,$dKey];
        }
    }
}
