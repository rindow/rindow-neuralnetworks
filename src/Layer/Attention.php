<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Variable;

class Attention extends AbstractAttentionLayer
{
    protected bool $useScale;
    protected bool $doNotExpandMask;
    protected NDArray $scale;
    protected NDArray $dScale;
    /** @var array<bool> $unbackpropagatables */
    protected ?array $unbackpropagatables = null;
    protected float $mask_exp = -1e9;

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
        ?array $input_shapes=null,
        ?bool $use_scale=null,
        ?bool $do_not_expand_mask=null,
        ?string $name=null,
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
        $this->initName($name,'attention');
        if($this->useScale) {
            $this->scale = $K->array(1.0);
            $this->dScale = $K->array(0.0);
            $this->allocateWeights(['scale']);
        }
        if($backend->deviceType()=='PHP') {
            $this->mask_exp = -1e99;
        }
    }

    public function build(mixed $variables=null, ?array $sampleWeights=null) : void
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

    public function reverseSyncWeightVariables() : void
    {
        if($this->useScale) {
            $this->scale = $this->weights[0]->value();
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
     * @param array<Variable> $inputs
     * @param array<Variable> $mask
     * @return array<Variable>|Variable
     */
    public function forward(
        array $inputs, 
        Variable|bool|null $training=null, 
        Variable|bool|null $returnAttentionScores=null,
        ?array $mask=null,
        ) : Variable|array
    {
        //$outputs = null;
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
        } else {
            if(count($inputs)<2) {
                throw new InvalidArgumentException('inputs must be a list of two or more NDArrays.');
            }
            $rawMask = $this->retrieveMultiMasks($rawInputs);
            //[$mask,$rawMask] = $this->packAndUnpackVariables($this->backend,$rawMask,unbackpropagatable:true);
            //$options['queryMask'] = $mask[0];
            //$options['valueMask'] = $mask[1];
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
                $rawOutputs[0] = $this->makeSingleMaskedValue($rawInputs[0], $rawOutputs[0]);
                $rawOutputs[1] = $this->makeSingleMaskedValue($rawInputs[0], $rawOutputs[1]);
                $this->assertOutputShape($rawOutputs[0],'forward');
                $this->assertScoresShape($rawOutputs[1],'forward');
            } else {
                $rawOutputs = $this->makeSingleMaskedValue($rawInputs[0], $rawOutputs);
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
     * @param array<NDArray|null> $mask
     * @return NDArray|array<NDArray>
     */
    protected function call(
        array $inputs,
        ?bool $training=null,
        ?bool $returnAttentionScores=null,
        ?array $mask=null,
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
        $scores = $K->matmul($query, $key, transB:true);
        
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
            $queryMask = $mask[0] ?? null;
            $valueMask = $mask[1] ?? null;
        }
        if($valueMask) {
            $mask_exp = $this->mask_exp;
            //if($valueMask->dtype()==NDArray::bool || $K->isInt($valueMask)) {
            //    $valueMask = $K->cast($valueMask,$scores->dtype());
            //}
            if(!$this->doNotExpandMask) { // Broadcasting 
                if($valueMask->ndim()<2) {
                    throw new InvalidArgumentException('value mask must be 2D array.');
                }
                // scores = [batch_size, Tq, Tv]
                // valueMask = [batch_size, Tv]
                $K->update_masking($scores,$valueMask,fill:$mask_exp,batchDims:-2,axis:-1);
            } else { // No Broadcasting 
                // scores += (-1e9*valueMask)
                $valueMask = $this->expandMask($valueMask,$scores);
                $K->update_masking($scores,$valueMask,fill:$mask_exp);
            }
        }
        // weights = softmax(scores)
        $attentionWeight = $K->softmax($scores);

        // vector = weights * value
        // scores = [batch_size, Tq, Tv]
        // value  = [batch_size, Tv, dim]
        // vector = [batch_size, Tq, dim]
        $contextVector = $K->matmul($attentionWeight, $value);

        if($queryMask) {
            //if($K->isFloat($queryMask)) {
            //    $queryMask = $K->greater($queryMask,0.5);
            //} else {
            //    $queryMask = $K->cast($queryMask,$contextVector->dtype());
            //}
            if(!$this->doNotExpandMask) { // Broadcasting
                // queryMask = [batch_size, Tq]
                // vector    = [batch_size, Tq, dim]
                if($queryMask->ndim()<2) {
                    throw new InvalidArgumentException('query mask must be 2D array.');
                }
                $K->update_masking($contextVector,$queryMask,batchDims:$queryMask->ndim(),axis:$contextVector->ndim());
            } else { // No Broadcasting 
                $queryMask = $this->expandMask($queryMask,$contextVector);
                $contextVector = $K->update_masking($contextVector, $queryMask);
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
            $queryMask = $container->queryMask;
            if(!$this->doNotExpandMask) { // Broadcasting 
                // queryMask = [batch_size, Tq]
                // vector = [batch_size, Tq, dim]
                $dOutputs = $K->masking($queryMask,$dOutputs,batchDims:$queryMask->ndim(),axis:$dOutputs->ndim());
            } else {
                $dOutputs = $K->masking($queryMask,$dOutputs);
            }
        }

        $dAttentionWeight = $K->matmul($dOutputs,$container->value,transA:false,transB:true);
        $dValue = $K->matmul($container->attentionWeight,$dOutputs,transA:true,transB:false);

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

        $dQuery = $K->matmul($dScores,$container->key,transA:false,transB:false);
        $dKey = $K->matmul($dScores,$container->query,transA:true,transB:false);

        if($container->sameKey) {
            $K->update_add($dValue,$dKey);
            return [$dQuery,$dValue];
        } else {
            return [$dQuery,$dValue,$dKey];
        }
    }

}
