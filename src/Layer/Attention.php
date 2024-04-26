<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use ArrayAccess;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Gradient\Core\GradientTape;
use Rindow\NeuralNetworks\Gradient\Core\GradientUtils;

class Attention extends AbstractLayerBase
{
    use GenericUtils;
    use GradientUtils;
    protected bool $useScale;
    protected bool $doNotExpandMask;
    protected NDArray $scale;
    protected NDArray $dScale;
    /** @var array<int> $scoresShape */
    protected $scoresShape;
    /** @var array<bool> $unbackpropagatables */
    protected ?array $unbackpropagatables = null;

    //protected $returnAttentionScores;

    //protected $query;
    //protected $value;
    //protected $key;
    //protected $attentionWeight;

    /**
     * @param array<array<int>> $input_shapes
     */
    public function __construct(
        object $backend,
        array $input_shapes=null,
        bool $use_scale=null,
        bool $do_not_expand_mask=null,
        string $name=null,
    )
    {
        // defaults
        $input_shapes = $input_shapes ?? null;
        $name = $name ?? null;
        $use_scale = $use_scale ?? false;
        $do_not_expand_mask = $do_not_expand_mask ?? false;

        parent::__construct($backend);
        $K = $backend;
        $this->inputShape = $input_shapes;
        $this->useScale = $use_scale;
        $this->doNotExpandMask = $do_not_expand_mask;
        if($this->useScale) {
            $this->scale = $K->array(1.0);
            $this->dScale = $K->array(0.0);
            $this->allocateWeights(1);
        }
        $this->initName($name,'attention');
    }

    public function build(mixed $variables=null, array $sampleWeights=null) : void
    {
        $K = $this->backend;
        $inputShapes = $this->normalizeInputShapes($variables);
        if(count($inputShapes)!=2&&count($inputShapes)!=3) {
            throw new InvalidArgumentException('num of inputs must be 2 or 3: inputs is '.count($inputShapes));
        }
        foreach ($inputShapes as $idx => $shape) {
            if(!is_array($shape)||count($shape)<2) {
                $type = '['.implode(',',$shape).']';
                throw new InvalidArgumentException('input_shapes must be the list of shape: '.$type.' included in #'.$idx.'.');
            }
        }
        $query = $inputShapes[0];  // Query
        $dim = array_pop($query);
        $tq  = array_pop($query);
        $value = $inputShapes[1]; // Value
        $tdim = array_pop($value);
        $tv =   array_pop($value);
        if($dim!=$tdim || $query!=$value) {
            throw new InvalidArgumentException('Unmatch query shape and value shape:'.
            '['.implode(',',$inputShapes[0]).'],['.implode(',',$inputShapes[1]).']');
        }
        if(count($inputShapes)==3) {
            if($inputShapes[1]!=$inputShapes[2]) {
                throw new InvalidArgumentException('value shape and key shape must be same.');
            }
        }
        $this->outputShape = array_merge($query,[$tq,$dim]);
        $this->scoresShape = array_merge($query,[$tq,$tv]);
        $this->syncWeightVariables();
    }

    public function getParams() : array
    {
        if($this->useScale) {
            return [$this->scale];
        } else {
            return [];
        }
    }

    public function getGrads() : array
    {
        if($this->useScale) {
            return [$this->dScale];
        } else {
            return [];
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

    /**
     * @param array<NDArray> $inputs
     */
    protected function assertInputShapes(array $inputs,string $direction) : void
    {
        if(!$this->shapeInspection)
            return;
        if($this->inputShape===null) {
            throw new InvalidArgumentException('Uninitialized input shape in '.$this->name.':'.$direction);
        }
        if(count($inputs)!=2 && count($inputs)!=3) {
            throw new InvalidArgumentException('Must have 2 or 3 arguments in '.$this->name.':'.$direction);
        }
        //$tq = $this->inputShape[0][0];
        //$dim = $this->inputShape[0][1];
        //$tv = $this->inputShape[1][0];
        $qshape = $inputs[0]->shape();
        $batchNum = array_shift($qshape);
        $vshape = $inputs[1]->shape();
        $vbatchNum = array_shift($vshape);
        if($batchNum!=$vbatchNum) {
            throw new InvalidArgumentException('Unmatch batch size of query and value: '.
                "query=[$batchNum,".implode(',',$qshape)."],".
                "value=[$vbatchNum,".implode(',',$vshape)."]".
                "in ".$this->name.':'.$direction);
        }
        if($this->inputShape[0]!=$qshape){
            throw new InvalidArgumentException('Unmatch query shape '.
                ' [b,'.implode(',',$this->inputShape[0]).'] NDArray.'.
                ' ['.$batchNum.','.implode(',',$qshape).'] given in '.$this->name.':'.$direction);
        }
        if($this->inputShape[1]!=$vshape){
            throw new InvalidArgumentException('Unmatch value shape '.
                ' [b,'.implode(',',$this->inputShape[1]).'] NDArray.'.
                ' ['.$vbatchNum.','.implode(',',$vshape).'] given in '.$this->name.':'.$direction);
        }
        if(count($inputs)==3) {
            $kshape = $inputs[2]->shape();
            $kbatchNum = array_shift($kshape);
            if($vbatchNum!=$kbatchNum) {
                throw new InvalidArgumentException('Unmatch batch size of value and key: '.
                    "query=[$vbatchNum,".implode(',',$vshape)."],".
                    "value=[$kbatchNum,".implode(',',$kshape)."]".
                    "in ".$this->name.':'.$direction);
            }
            if($kshape!=$vshape){
                throw new InvalidArgumentException('Unmatch value shape and key shape.:'.
                    ' ['.implode(',',$vshape).'],['.implode(',',$kshape).'] in '.$this->name.':'.$direction);
            }
        }
    }

    protected function assertScoresShape(NDArray $scores,string $direction) : void
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
            throw new InvalidArgumentException('unmatch scores shape: '.$shape.', must be '.$scoresShape.' in '.$this->name.':'.$direction);
        }
    }

    /**
     * @param array<NDArray> $dOutputs
     * @param ArrayAccess<object,object> $grads
     * @param array<NDArray> $oidsToCollect
     * @return array<NDArray>
     */
    public function backward(
        array $dOutputs,
        ArrayAccess $grads=null,
        array $oidsToCollect=null
        ) : array
    {
        if(count($dOutputs)!=1&&count($dOutputs)!=2) {
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

    protected function expandMask(NDArray $sourceMask,NDArray $target) : NDArray
    {
        $K = $this->backend;
        $mask = $sourceMask;
        $maskShape = $mask->shape();
        $targetShape = $target->shape();
        foreach (array_map(null,$maskShape,$targetShape) as $axis => [$mT,$T]) {
            if($mT==1 && $T!=1) {
                $mask = $K->repeat($mask,$T,axis:$axis,keepdims:true);
            } elseif($mT!=$T) {
                throw new InvalidArgumentException('Incompatible shapes for broadcasting: '.
                    '['.implode(',',$sourceMask->shape()).'] vs. ['.implode(',',$target->shape()).']');
            }
        }
        return $mask;
    }

    /**
     * @param array<NDArray> $inputs
     * @param array{NDArray,NDArray} $mask
     * @return NDArray|array<NDArray>
     */
    protected function call(
        array $inputs,
        bool $training=null,
        bool $returnAttentionScores=null,
        array $mask=null,
        ) : NDArray|array
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
        //
        // query  = [batch_size, Tq, dim]
        // key    = [batch_size, Tv, dim]
        // scores = [batch_size, Tq, Tv]
        $scores = $K->matmul($query, $key, null, $tranB=true);
        
        $container->useScale = $this->useScale;
        if($this->useScale) {
            // scores = scores / sqrt(qk) 
            $scale = $K->scalar($this->scale);
            $scores = $K->update_scale($scores,$scale);
            $container->scale = $scale;
            $container->scores = $K->copy($scores);
        }
        $queryMask = null;
        $valueMask = null;
        if($mask) {
            [$queryMask,$valueMask] = $mask;
        }
        if($valueMask) {
            if($valueMask->dtype()==NDArray::bool || $K->isInt($valueMask)) {
                $valueMask = $K->cast($valueMask,$scores->dtype());
            }
            $valueMask = $K->less($valueMask,0.5);              // (mask<0.5)>1.0 , (mask>0.5)=>0.0
            // scores += (-1e9*valueMask)
            if(!$this->doNotExpandMask) { // Broadcasting 
                // scores = [batch_size, Tq, Tv]
                // valueMask = [batch_size, Tv]
                $scoresShape = $scores->shape();
                $Tv = array_pop($scoresShape);
                $Tq = array_pop($scoresShape);
                $maskShape = $valueMask->shape();
                $mTv = array_pop($maskShape);
                if($maskShape!=$scoresShape||$Tv!=$mTv) {
                    throw new InvalidArgumentException('unmatch inputs and queryMask.'.
                    ' scores:['.implode(',',$scores->shape()).']'.
                    ' given mask:['.implode(',',$valueMask->shape()).']');
                }
                // scores = [batch_size, Tq, Tv]
                // valueMask = [batch_size, Tv] =repeat=> [batch_size, Tq, Tv]
                $valueMask = $K->repeat($valueMask,$Tq,axis:-1);
            } else { // No Broadcasting 
                // scores += (-1e9*valueMask)
                $valueMask = $this->expandMask($valueMask,$scores);
            }
            $K->update_add($scores,$valueMask,alpha:-1e9);
        }
        // weights = softmax(scores)
        $attentionWeight = $K->softmax($scores);

        // vector = weights * value
        // scores = [batch_size, Tq, Tv]
        // value  = [batch_size, Tv, dim]
        // vector = [batch_size, Tq, dim]
        $contextVector = $K->matmul($attentionWeight, $value);

        if($queryMask) {
            if($K->isFloat($queryMask)) {
                $queryMask = $K->greater($queryMask,0.5);
            } else {
                $queryMask = $K->cast($queryMask,$contextVector->dtype());
            }
            if(!$this->doNotExpandMask) { // Broadcasting 
                // queryMask = [batch_size, Tq]
                // vector = [batch_size, Tq, dim] => [dim, batch_size, Tq]
                $shape = $contextVector->shape();
                $orgShape = $shape;
                $dim = array_pop($shape);
                if($queryMask->shape()!=$shape) {
                    throw new InvalidArgumentException('unmatch inputs and queryMask.'.
                    ' contextVector:['.implode(',',$contextVector->shape()).']'.
                    ' given mask:['.implode(',',$queryMask->shape()).']');
                }
                $Tq = array_pop($shape);
                $batchSize = (int)array_product($shape);
                $queryMask = $queryMask->reshape([(int)array_product($queryMask->shape())]);
                $contextVector = $contextVector->reshape([$batchSize*$Tq, $dim]);
                $contextVector = $K->update_mul($contextVector, $queryMask, trans:true);
                $contextVector = $contextVector->reshape($orgShape);
            } else { // No Broadcasting 
                $queryMask = $this->expandMask($queryMask,$contextVector);
                $contextVector = $K->update_mul($contextVector, $queryMask);
            }
            $container->queryMask = $queryMask;
        }

        $container->value = $value;
        $container->attentionWeight = $attentionWeight;
        $container->query = $query;
        $container->key = $key;
        if($returnAttentionScores) {
            $this->unbackpropagatables = [false,true];
            return [$contextVector,$attentionWeight];
        } else {
            return $contextVector;
        }
    }

    /**
     * @return array<NDArray>
     */
    protected function differentiate(NDArray $dOutputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        // forward:
        //   vector = weights (*) value
        // backward:
        //   dWeights = dVector (*) value^T
        //   dValue   = weights^T (*) dVector

        if(isset($container->queryMask)) {
            if(!$this->doNotExpandMask) { // Broadcasting 
                // queryMask = [batch_size*Tq]
                // vector = [batch_size, Tq, dim] => [dim, batch_size, Tq]
                $batchShape = $dOutputs->shape();
                $dim = array_pop($batchShape);
                $Tq = array_pop($batchShape);
                $batchSize = (int)array_product($batchShape);
                $dOutputs = $K->copy($dOutputs->reshape([$batchSize*$Tq, $dim]));
                $dOutputs = $K->update_mul($dOutputs, $container->queryMask, trans:true);
                $dOutputs = $dOutputs->reshape([$batchSize, $Tq, $dim]);
            } else {
                $dOutputs = $K->copy($dOutputs);
                $dOutputs = $K->update_mul($dOutputs, $container->queryMask);
            }
        }

        $dAttentionWeight = $K->matmul($dOutputs,$container->value,$transA=false,$transB=true);
        $dValue = $K->matmul($container->attentionWeight,$dOutputs,$transA=true,$transB=false);

        $dScores = $K->dSoftmax($dAttentionWeight,$container->attentionWeight);

        // valueMask is dAdd so it is passed through.

        if($container->useScale) {
            // dScale  = sum(dScales * scales)
            // dScores = dScores * scale 
            $dScale = $K->sum($K->mul($dScores,$container->scores));
            if(is_scalar($dScale)) {
                $dScale = $K->array($dScale);
            }
            $K->copy($dScale,$this->dScale);
            $K->update_scale($dScores,$container->scale);
        }

        $dQuery = $K->matmul($dScores,$container->key,$transA=false,$transB=false);
        $dKey = $K->matmul($dScores,$container->query,$transA=true,$transB=false);

        if($container->sameKey) {
            $K->update_add($dValue,$dKey);
            return [$dQuery,$dValue];
        } else {
            return [$dQuery,$dValue,$dKey];
        }
    }

    /**
     * @param array<string,mixed> $options
     */
    protected function numOfOutputs(?array $options) : int
    {
        if($options['returnAttentionScores'] ?? false) {
            return 2;
        }
        return 1;
    }

    /**
     * @return array<Variable>|Variable
     */
    final public function __invoke(mixed ...$args) : array|NDArray
    {
        return $this->forward(...$args);
    }

    /**
     * @param array<Variable> $inputs
     * @param array<Variable> $mask
     * @return array<Variable>|Variable
     */
    public function forward(
        array $inputs, 
        Variable|bool $training=null, 
        Variable|bool $returnAttentionScores=null,
        array $mask=null,
        )
    {
        //$outputs = null;
        if(!is_array($inputs)) {
            throw new InvalidArgumentException('inputs must be list of Variable');
        }
        [$inputs,$rawInputs]     = $this->packAndUnpackVariables($this->backend,$inputs);
        $options = [];
        [$training,$rawTraining] = $this->packAndUnpackVariable($this->backend,$training,unbackpropagatable:true);
        [$returnAttentionScores,$rawReturnAttentionScores] = $this->packAndUnpackVariable($this->backend,$returnAttentionScores,unbackpropagatable:true);
        $options['training'] = $training;
        $options['returnAttentionScores'] = $returnAttentionScores;
        $rawMask = null;
        if($mask) {
            if(count($mask)!=2) {
                throw new InvalidArgumentException('mask must be list of the two of masks as queryMask and valueMask');
            }
            [$mask,$rawMask] = $this->packAndUnpackVariables($this->backend,$mask,unbackpropagatable:true);
            $options['queryMask'] = $mask[0];
            $options['valueMask'] = $mask[1];
        }
        if(!$this->built) {
            $this->build($inputs);
            $this->built = true;
        }
        $options = $this->cleanNullValue($options);
        
        $numOfOutputs = $this->numOfOutputs($options);
        $session = $this->preGradientProcessOnSession($inputs,$options);
        $session->begin();
        try {
            $this->assertInputShapes($rawInputs,'forward');
            $this->unbackpropagatables = null;
            $rawOutputs = $this->call(
                $rawInputs, 
                training:$rawTraining, 
                returnAttentionScores:$rawReturnAttentionScores,
                mask:$rawMask,
                );
            if($returnAttentionScores){
                $this->assertOutputShape($rawOutputs[0],'forward');
                $this->assertScoresShape($rawOutputs[1],'forward');
            } else {
                $this->assertOutputShape($rawOutputs,'forward');
            }
        } finally{
            $session->end();
        }
        if($numOfOutputs==1) {
            $rawOutputs = [$rawOutputs];
        }
        $outputs = $this->postGradientProcessOnSession(
            $this->backend, $session,$inputs,
            $rawOutputs,$this->unbackpropagatables);
        if($numOfOutputs==1) {
            return $outputs[0];
        } else {
            return $outputs;
        }
    }

    /**
     * Call from SessionFunc in compiled graph
     * @param array<NDArray> $inputs
     * @param array<string,mixed> $options
     * @return array<NDArray>
     */
    public function _rawCall(array $inputs,array $options) : array
    {
        $training = $options['training'] ?? null;
        $queryMask = $options['queryMask'] ?? null;
        $valueMask = $options['valueMask'] ?? null;
        $mask = null;
        if($queryMask) {
            $mask = [$queryMask,$valueMask];
        }
        $returnAttentionScores = $options['returnAttentionScores'] ?? null;
        $outputs = $this->call(
            $inputs,
            training:$training,
            returnAttentionScores:$returnAttentionScores,
            mask:$mask,
        );
        if(!is_array($outputs)) {
            $outputs = [$outputs];
        }
        return $outputs;
    }

}
