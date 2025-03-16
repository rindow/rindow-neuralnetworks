<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use ArrayAccess;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Gradient\Core\GradientTape;
use Rindow\NeuralNetworks\Gradient\Core\GradientUtils;
use Rindow\NeuralNetworks\Layer\Activation;
use Rindow\NeuralNetworks\Layer\Dropout;

class MultiHeadAttention extends AbstractAttentionLayer
{
    protected bool $supportsMasking;
    protected int $numHeads;
    protected int $keyDim;
    protected int $valueDim;
    protected float $dropout;
    protected bool $useBias;
    protected float $inverse_sqrt_key_dim;
    protected mixed $kernelInitializer;
    protected mixed $biasInitializer;
    protected ?string $kernelInitializerName;
    protected ?string $biasInitializerName;
    /** @var array<int> $attention_axes */
    protected ?array $attention_axes;
    /** @var array<int> $query_feature_shape */
    protected array $query_feature_shape;
    /** @var array<int> $key_feature_shape */
    protected array $key_feature_shape;
    /** @var array<int> $value_feature_shape */
    protected array $value_feature_shape;
    protected Layer $query_dense;
    protected Layer $key_dense;
    protected Layer $value_dense;
    protected string $dot_product_equation = 'aecd,abcd->acbe';
    protected string $backward_dot_product_key_equation = 'acbe,abcd->aecd';
    protected string $backward_dot_product_query_equation = 'acbe,aecd->abcd';
    protected string $combine_equation = 'acbe,aecd->abcd';
    protected string $backward_combine_scores_equation = 'abcd,aecd->acbe';
    protected string $backward_combine_value_equation = 'abcd,acbe->aecd';
    protected ?Layer $dropout_layer=null;
    protected Layer $output_dense;
    protected bool $useScale;
    protected bool $doNotExpandMask;
    protected NDArray $scale;
    protected NDArray $dScale;
    /** @var array<bool> $unbackpropagatables */
    protected ?array $unbackpropagatables = null;
    protected float $mask_exp = -1e9;

    /**
     * @param array<array<int>> $input_shapes
     * @param int|array<int> $attention_axes
     */
    public function __construct(
        object $backend,
        ?int $num_heads,
        ?int $key_dim,
        ?int $value_dim=null,
        ?float $dropout=null,
        ?bool $use_bias=null,
        ?array $input_shapes=null,
        int|array|null $attention_axes=null,
        string|object|null $kernel_initializer=null,
        string|object|null $bias_initializer=null,
        ?string $name=null,
    )
    {
        $value_dim ??= $key_dim;
        $dropout ??= 0.0;
        $use_bias ??= true;
        $kernel_initializer ??= 'glorot_uniform';
        $bias_initializer ??= 'zeros';

        parent::__construct($backend);
        $K = $backend;
        $this->supportsMasking = true;
        $this->numHeads = $num_heads;
        $this->keyDim = $key_dim;
        $this->inverse_sqrt_key_dim = 0.0;
        $this->valueDim = $value_dim;
        $this->dropout = $dropout;
        $this->useBias = $use_bias;
        $this->inputShape = $input_shapes;
        if($attention_axes!==null) {
            if(is_int($attention_axes)) {
                $attention_axes = [$attention_axes];
            }
        }
        $this->kernelInitializer = $K->getInitializer($kernel_initializer);
        $this->biasInitializer   = $K->getInitializer($bias_initializer);
        $this->kernelInitializerName = $this->toStringName($kernel_initializer);
        $this->biasInitializerName = $this->toStringName($bias_initializer);
        $this->attention_axes = $attention_axes;
        if($backend->deviceType()=='PHP') {
            $this->mask_exp = -1e99;
        }
        $this->initName($name,'multiheadattention');
    }
    
    /**
     * query_shape: Shape of the query tensor.
     * value_shape: Shape of the value tensor.
     * key: Optional shape of the key tensor.
     */
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
        $query_shape = $inputShapes[0];  // Query
        $value_shape = $inputShapes[1];  // Value
        if(count($inputShapes)==3) {
            if($inputShapes[1]!=$inputShapes[2]) {
                throw new InvalidArgumentException('value shape and key shape must be same.');
            }
            $key_shape = $inputShapes[2]; // Key;
        }
        $key_shape ??= $value_shape;

        array_unshift($query_shape,1); // (Batch, (Tq), FeatureQ)
        array_unshift($value_shape,1); // (Batch, (Tv), FeatureV)
        array_unshift($key_shape,1);   // (Batch, (Tv), FeatureK)

        $query_rank = count($query_shape);                      // rank = 3+?
        $value_rank = count($value_shape);                      // rank = 3+?
        $key_rank = count($key_shape);                          // rank = 3+?
        if($query_rank!=$value_rank||$query_rank!=$key_rank) {
            throw new InvalidArgumentException('query, value, key must have same rank.');
        }
        if($this->attention_axes==null) {
            $this->attention_axes = range(1, $query_rank-2);
        }
        $common_args = [
            'kernel_initializer'=>$this->kernelInitializerName,
            'bias_initializer'=>$this->biasInitializerName,
            'use_bias'=>$this->useBias,
        ];

        //
        // query dense
        //
        // (B.Tq.Fq),(Fq.H.Dk)->(B.Tq.H.Dk)         // gemm(batches.Feature,Feature.units) => batches.units
        // equation:'abc,cde->abde'
        [$batch,$Tq,$Fq,$units,$dense_input_shape] = $this->build_dense_args(
            $query_shape,[$this->numHeads, $this->keyDim],
        );
        $this->query_feature_shape = $Fq;           // kernel(Feature.units) , bias(units)
        $this->query_dense = new Dense(             // Dense(inputs(batches.Feature),units)
            $this->backend,                         //     units       : (numHeads.keyDim)
            $units,                                 //     input_shape : ((Tq),Feature)
            input_shape:$dense_input_shape,         //     output_shape: ((Tq),(numHeads.keyDim))
            name:"query.{$this->name}",             //     use_bias
            kernel_initializer:$this->kernelInitializerName,
            bias_initializer:$this->biasInitializerName,
            use_bias:$this->useBias,
        );
        $sampleW = null;
        if($sampleWeights!==null) {
            $sampleW = [];
            $sampleW[] = array_shift($sampleWeights);       // query_dense/kernel
            if($this->useBias) {
                $sampleW[] = array_shift($sampleWeights);   // query_dense/bias
            }
        }
        $this->query_dense->build($dense_input_shape,sampleWeights:$sampleW);

        //
        // key dense
        //
        // (B.Tv.Fk),(Fk.H.Dk)->(B.Tv.H.Dk)         // gemm(batches.Feature,Feature.units) => batches.units
        // equation:'abc,cde->abde'
        [$batch,$Tk,$Fk,$units,$dense_input_shape] = $this->build_dense_args(
            $key_shape,[$this->numHeads, $this->keyDim],
        );
        $this->key_feature_shape = $Fk;             // kernel(Feature.units) , bias(units)
        $this->key_dense = new Dense(               // Dense(inputs(batches.Feature),units)
            $this->backend,                         //     units       : (numHeads.keyDim)
            $units,                                 //     input_shape : ((Tq),Feature)
            input_shape:$dense_input_shape,         //     output_shape: ((Tq),(numHeads.keyDim))
            name:"key.{$this->name}",               //     use_bias
            kernel_initializer:$this->kernelInitializerName,
            bias_initializer:$this->biasInitializerName,
            use_bias:$this->useBias,
        );
        $sampleW = null;
        if($sampleWeights!==null) {
            $sampleW = [];
            $sampleW[] = array_shift($sampleWeights);       // key_dense/kernel
            if($this->useBias) {
                $sampleW[] = array_shift($sampleWeights);   // key_dense/bias
            }
        }
        $this->key_dense->build($dense_input_shape,sampleWeights:$sampleW);

        //
        // value dense
        //
        // (B.Tv.Fv),(Fv.H.Dv)->(B.Tv.H.Dv)        // gemm(batches.Feature,Feature.units) => batches.units
        // equation:'abc,cde->abde'
        [$batch,$Tv,$Fv,$units,$dense_input_shape] = $this->build_dense_args(
            $value_shape,[$this->numHeads, $this->valueDim],
        );
        $this->value_feature_shape = $Fv;           // kernel(Feature.units) , bias(units)
        $this->value_dense = new Dense(             // Dense(inputs(batches.Feature),units)
            $this->backend,                         //     units       : (numHeads.keyDim)
            $units,                                 //     input_shape : ((Tq),Feature)
            input_shape:$dense_input_shape,         //     output_shape: ((Tq),(numHeads.keyDim))
            name:"value.{$this->name}",             //     use_bias
            kernel_initializer:$this->kernelInitializerName,
            bias_initializer:$this->biasInitializerName,
            use_bias:$this->useBias,
        );                                          
        $output_rank = 1+count($Tv)+1+1; // (B,(Tv),Head,KeyDim)
        $sampleW = null;
        if($sampleWeights!==null) {
            $sampleW = [];
            $sampleW[] = array_shift($sampleWeights);       // value_dense/kernel
            if($this->useBias) {
                $sampleW[] = array_shift($sampleWeights);   // value_dense/bias
            }
        }
        $this->value_dense->build($dense_input_shape,sampleWeights:$sampleW);

        // 
        // scale query
        //
        $this->inverse_sqrt_key_dim = 1.0 / sqrt($this->keyDim);

        //
        // attention scores
        //
        // scores = einsum(equation, key, query)
        // key:    ((Batch), (Tv), numHeads, keyDim)
        // query:  ((Batch), (Tq), numHeads, keyDim)
        // scores: ((Batch), numHeads, (Tq), (Tv))
        //
        // dot_product_equation                = 'aecd,abcd->acbe';
        // backward_dot_product_key_equation   = 'acbe,abcd->aecd';
        // backward_dot_product_query_equation = 'acbe,aecd->abcd';

        //
        // dropout
        //
        if($this->dropout>0.0) {
            $this->dropout_layer = new Dropout(
                $this->backend,
                rate:$this->dropout
            );
        }

        //
        // attention outputs
        //
        // output = einsum(equation,scores,value)
        // scores: ((Batch), numHeads, (Tq), (Tv))
        // value:  ((Batch), (Tv), numHeads, valueDim)
        // output: ((Batch), (Tq), numHeads, valueDim)
        // combine_equation = 'acbe,aecd->abcd';
        // backward_combine_scores_equation = 'abcd,aecd->acbe';
        // backward_combine_value_equation = 'abcd,acbe->aecd';

        //
        // output dense
        //
        // input:   ((Batch, (Tq)), (numHeads, valueDim))
        // kernel:  ((numHeads, valueDim), Fq)
        // output:  ((Batch, (Tq)), Fq)
        // equation: ab.cd,cd.e->ab.e    =>  gemm(x,y)
        $output_dense_input_shape = array_merge(
            [$batch],$Tq,[$this->numHeads, $this->valueDim]
        );
        [$batch,$To,$Fo,$units,$dense_input_shape] = $this->build_dense_args(
            $output_dense_input_shape, $Fq,
        );
        $this->output_dense = new Dense(
            $this->backend,
            $units,
            input_shape: $dense_input_shape, 
            name: "output.{$this->name}",
            kernel_initializer:$this->kernelInitializerName,
            bias_initializer:$this->biasInitializerName,
            use_bias:$this->useBias,
        );
        $sampleW = null;
        if($sampleWeights!==null) {
            $sampleW = [];
            $sampleW[] = array_shift($sampleWeights);       // output_dense/kernel
            if($this->useBias) {
                $sampleW[] = array_shift($sampleWeights);   // output_dense/bias
            }
        }
        $this->output_dense->build($dense_input_shape,sampleWeights:$sampleW);

        //
        // output shape
        //
        // outputs: (Batch, (Tq), (Fq))
        $this->outputShape = array_merge($Tq,$Fq);
        // scores: (Batch, numHeads, (Tq), (Tv))
        $n_attn_axes = count($this->attention_axes);
        $querySeq = array_slice($this->inputShape[0],0,$n_attn_axes);
        $keySeq = array_slice($this->inputShape[1],0,$n_attn_axes);
        $this->scoresShape =array_merge([$this->numHeads],$querySeq,$keySeq);
    }

    public function weights() : array
    {
        return array_merge(
            $this->query_dense->weights(),
            $this->key_dense->weights(),
            $this->value_dense->weights(),
            $this->output_dense->weights(),
        );
    }

    public function getParams() : array
    {
        return array_merge(
            $this->query_dense->getParams(),
            $this->key_dense->getParams(),
            $this->value_dense->getParams(),
            $this->output_dense->getParams(),
        );
    }

    public function getGrads() : array
    {
        return array_merge(
            $this->query_dense->getGrads(),
            $this->key_dense->getGrads(),
            $this->value_dense->getGrads(),
            $this->output_dense->getGrads(),
        );
    }

    public function reverseSyncWeightVariables() : void
    {
        $this->query_dense->reverseSyncWeightVariables();
        $this->key_dense->reverseSyncWeightVariables();
        $this->value_dense->reverseSyncWeightVariables();
        $this->output_dense->reverseSyncWeightVariables();
    }

    public function getConfig() : array
    {
        return [
            'num_heads'=>$this->numHeads,
            'key_dim'=>$this->keyDim,
            'options' => [
                'value_dim'=>$this->valueDim,
                'dropout'=>$this->dropout,
                'use_bias'=>$this->useBias,
                'attention_axes'=>$this->attention_axes,
                'kernel_initializer' => $this->kernelInitializerName,
                'bias_initializer' => $this->biasInitializerName,
            ],
        ];
    }

    /**
     * @return array{NDArray,NDArray,NDArray}
     */
    private function compute_attention(
        NDArray $query,
        NDArray $key,
        NDArray $value,
        ?NDArray $query_mask=null,
        ?NDArray $value_mask=null,
        ?NDArray $key_mask=null,
        ?NDArray $attention_mask=null,
        ?bool $useCausalMask=null,
        ?bool $training=null
    ) : array
    {
        $K = $this->backend;
        $mo = $this->backend->localMatrixOperator();

        // Note: Applying scalar multiply at the smaller end of einsum improves
        // XLA performance, but may introduce slight numeric differences in
        // the Transformer attention head.    
        $scaled_query = $K->scale(
            $this->inverse_sqrt_key_dim,
            $query,
        );

        //
        // dot_product: (B,(Tk),H,Dk),(B,(Tq),H,Dk) -> (B,H,(Tq),(Tk))
        //              'aecd,abcd->acbe'
        // scores = einsum(equation, key, query)
        //
        [$key_input_shape,$query_input_shape,$score_output_shape] = 
            $this->make_forward_product_shape($key, $scaled_query);
        
        $attention_scores = $K->einsum4p1(
            $this->dot_product_equation,
            $key->reshape($key_input_shape),
            $scaled_query->reshape($query_input_shape),
        );
        $attention_scores = $attention_scores->reshape($score_output_shape);

        //
        // Apply a mask to the attention scores
        //
        // query_mask:          (B, (Tq)
        // value_mask:          (B, (Tv)
        // key_mask:            (B, (Tv)
        // causal_mask:         (B, (Tq), (Tv))
        // attention_masks:     (B, (Tq), (Tv))
        //      |
        //      V
        // attention_scores:    (B, H, (Tq), (Tv))
        //
        $attention_scores = $this->compute_masked_attention_scores(
            $attention_scores,
            $query,
            $value,
            query_mask:$query_mask,
            value_mask:$value_mask,
            key_mask:$key_mask,
            attention_mask:$attention_mask,
            useCausalMask:$useCausalMask,
        );

        //
        // attention softmax
        //
        // attention_scores:    (B, H, (Tq), (Tv))
        // Apply softmax along just Tv.
        // scores = softmax(scores)
        //
        $original_shape  = $attention_scores->shape();
        $value_seq_shape = $original_shape;
        $shape = array_splice($value_seq_shape,0,-count($this->attention_axes));
        $shape = array_merge($shape,[(int)array_product($value_seq_shape)]);
        $attention_scores = $K->softmax($attention_scores->reshape($shape));
        $attention_scores = $attention_scores->reshape($original_shape);
        //echo $K->localMatrixOperator()->toString($attention_scores,format:'%13.8e',indent:true)."\n";


        //
        // Apply dropout to attention scores
        // scores = dropout(scores)
        //
        // This is actually dropping out entire tokens to attend to, which might
        // seem a bit unusual, but is taken from the original Transformer paper.
        if($this->dropout_layer) {
            $final_attn_scores = $this->dropout_layer->_rawCall(
                [$attention_scores], ['training'=>$training]
            )[0];
        } else {
            $final_attn_scores = $attention_scores;
        }
    
        //
        // combine product: (B,H,(Tq),(Tk)),(B,(Tv),H,Dv) -> (B,(Tq),H,Dv)
        //                  'acbe,aecd->abcd'
        // output = einsum(equation,scores,value)
        //
        [$scores_input_shape,$value_input_shape,$combine_output_shape] = 
            $this->make_forward_combine_shape($final_attn_scores, $value);
        $attention_output = $K->einsum4p1(
            $this->combine_equation,
            $final_attn_scores->reshape($scores_input_shape),
            $value->reshape($value_input_shape),
        );
        $attention_output = $attention_output->reshape($combine_output_shape);

        return [$attention_output, $attention_scores, $scaled_query];
    }

    /**
     * @return array{NDArray,NDArray,NDArray}
     */
    private function compute_differntiate_attention(
        NDArray $dAttention_output,
        NDArray $scaled_query,
        NDArray $key,
        NDArray $value,
        NDArray $attention_output,
        NDArray $softmaxed_attention_scores,
        ?NDArray $attention_mask,
        ?bool $training,
    ) : array
    {
        $K = $this->backend;

        //
        // differential combine product
        //
        // dValue:
        //      (B,H,(Tq),(Tk)),(B,(Tq),H,Dv) -> (B,(Tv),H,Dv)
        //      'abcd,acbe->aecd'
        //      dValue = einsum(equation,dOutput,dScores)
        // dScores:
        //      (B,H,(Tq),(Tk)),(B,(Tv),H,Dv) -> (B,(Tq),H,Dv)
        //      'abcd,aecd->acbe'
        //      dScores = einsum(equation,dOutput,dValue)
        //
        [$scores_input_shape,$value_input_shape,$combine_output_shape] = 
            $this->make_backward_combine_shape($softmaxed_attention_scores, $value);
        $dValue = $K->einsum4p1(
            $this->backward_combine_value_equation,
            $dAttention_output->reshape($combine_output_shape),
            $softmaxed_attention_scores->reshape($scores_input_shape),
        );
        $dValue = $dValue->reshape($value->shape());
        $dSoftmaxedScores = $K->einsum4p1(
            $this->backward_combine_scores_equation,
            $dAttention_output->reshape($combine_output_shape),
            $value->reshape($value_input_shape),
        );
        $dSoftmaxedScores = $dSoftmaxedScores->reshape($softmaxed_attention_scores->shape());

        //
        // differential dropout
        //
        if($this->dropout_layer) {
            $dSoftmaxedScores = $this->dropout_layer->_rawDifferentiate([$dSoftmaxedScores])[0];
        }

        //
        // differential softmax
        //
        // scores:    (B, H, (Tq), (Tv))
        // Apply dSoftmax along just Tv.
        //
        $original_shape  = $softmaxed_attention_scores->shape();
        $value_seq_shape = $original_shape;
        $shape = array_splice($value_seq_shape,0,-count($this->attention_axes));
        $shape = array_merge($shape,[(int)array_product($value_seq_shape)]);
        $dScores = $K->dSoftmax(
            $dSoftmaxedScores->reshape($shape),
            $softmaxed_attention_scores->reshape($shape),
        );
        $dScores = $dScores->reshape($original_shape);


        //
        // differential dot product
        //
        // dKey:
        //      (B,H,(Tq),(Tk)),(B,(Tq),H,Dk) -> (B,(Tk),H,Dk)
        //      'acbe,abcd->aecd'
        //      dKey = einsum(equation,dScores,dQuery)
        // dQuery:
        //      (B,H,(Tq),(Tk)),(B,(Tk),H,Dk) -> (B,(Tq),H,Dk)
        //      'acbe,aecd->abcd'
        //      dQuery = einsum(equation,dScores,dKey)
        //
        [$key_input_shape,$query_input_shape,$score_output_shape] = 
            $this->make_backward_product_shape($key,$scaled_query);

        $dKey = $K->einsum4p1(
            $this->backward_dot_product_key_equation,
            $dScores->reshape($score_output_shape),
            $scaled_query->reshape($query_input_shape),
        );
        $dKey = $dKey->reshape($key->shape());

        $dScaledQuery = $K->einsum4p1(
            $this->backward_dot_product_query_equation,
            $dScores->reshape($score_output_shape),
            $key->reshape($key_input_shape),
        );
        $dScaledQuery = $dScaledQuery->reshape($scaled_query->shape());


        //
        // differential scale
        //
        $dQuery = $K->scale(
            $this->inverse_sqrt_key_dim,
            $dScaledQuery,
        );

        return [$dQuery, $dKey, $dValue];
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
        ?NDArray $attention_mask=null,
        Variable|bool|null $useCausalMask=null,
    )
    {
        [$inputs,$rawInputs]     = $this->packAndUnpackVariables($this->backend,$inputs);
        $options = [];
        [$training,$rawTraining] = $this->packAndUnpackVariable($this->backend,$training,unbackpropagatable:true);
        [$returnAttentionScores,$rawReturnAttentionScores] = $this->packAndUnpackVariable($this->backend,$returnAttentionScores,unbackpropagatable:true);
        [$useCausalMask,$rawUseCausalMask] = $this->packAndUnpackVariable($this->backend,$useCausalMask,unbackpropagatable:true);
        $options['training'] = $training;
        $options['returnAttentionScores'] = $returnAttentionScores;
        $options['useCausalMask'] = $useCausalMask;
        $rawMask = null;
        if($mask) {
            if(count($mask)<2) {
                throw new InvalidArgumentException('mask must be list of 2 or 3 of masks as queryMask and valueMask and keyMask');
            }
            [$mask,$rawMask] = $this->packAndUnpackVariables($this->backend,$mask,unbackpropagatable:true);
            $options['queryMask'] = $mask[0] ?? null;
            $options['valueMask'] = $mask[1] ?? null;
            $options['keyMask']   = $mask[2] ?? null;
        } else {
            $rawMask = $this->retrieveMultiMasks($rawInputs);
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
                attention_mask:$attention_mask,
                useCausalMask:$rawUseCausalMask,
            );
            if($returnAttentionScores){
                $this->assertOutputShape($rawOutputs[0],'forward');
                $this->assertScoresShape($rawOutputs[1],'forward');
                $rawOutputs[0] = $this->makeSingleMaskedValue($rawInputs[0], $rawOutputs[0]);
            } else {
                $this->assertOutputShape($rawOutputs,'forward');
                $rawOutputs = $this->makeSingleMaskedValue($rawInputs[0], $rawOutputs);
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
     * @param array<NDArray> $inputs
     * @param array<NDArray|null> $mask
     * @return array<NDArray>|NDArray
     */
    protected function call( 
        array $inputs,
        ?bool $training=null,
        ?bool $returnAttentionScores=null,
        ?array $mask=null,
        ?NDArray $attention_mask=null,
        ?bool $useCausalMask=null,
    ) : NDArray|array
    {
        $K = $this->backend;
        $mo = $this->backend->localMatrixOperator();

        $container = $this->container();
        $query = $inputs[0] ?? null;
        $value = $inputs[1] ?? null;
        $key = $inputs[2] ?? $value;
        if(count($inputs)==3) {
            $container->sameKey = false;
        } else {
            $container->sameKey = true;
        }
        $query_mask = $mask[0] ?? null;
        $value_mask = $mask[1] ?? null;
        $key_mask   = $mask[2] ?? null;

        //
        // query_dense: (B,(Tq),Fq),(Fq,H,Dk) -> (B,(Tq),H,Dk)
        //
        [$full_input_shape,$full_output_shape,$Fq] = $this->make_forward_dense_shape(
            $query, [$this->numHeads, $this->keyDim],
        );
        $query = $query->reshape($full_input_shape);
        $query = $this->query_dense->_rawCall([$query],['training'=>$training])[0];
        $query = $query->reshape($full_output_shape);

        //
        // key_dense: (B,(Tv),Fk),(Fk,H,Dk) -> (B,(Tv),H,Dk)
        //
        [$full_input_shape,$full_output_shape,$Fk] = $this->make_forward_dense_shape(
            $key, [$this->numHeads, $this->keyDim],
        );
        $key = $key->reshape($full_input_shape);
        $key = $this->key_dense->_rawCall([$key],['training'=>$training])[0];
        $key = $key->reshape($full_output_shape);

        //
        // value_dense: (B,(Tv),Fv),(Fv,H,Dv) -> (B,(Tv),H,Dv)
        //
        [$full_input_shape,$full_output_shape,$Fv] = $this->make_forward_dense_shape(
            $value, [$this->numHeads, $this->valueDim],
        );
        $value = $value->reshape($full_input_shape);
        $value = $this->value_dense->_rawCall([$value],['training'=>$training])[0];
        $value = $value->reshape($full_output_shape);

        //
        // dot_product:     (B,(Tv),H,Dk),(B,(Tq),H,Dk) -> (B,H,(Tq),(Tv))
        // combine_product: (B,H,(Tq),(Tv)),(B,(Tv),H,Dv) -> (B,(Tq),H,Dv)
        //
        [$attention_output, $attention_scores, $scaled_query] = $this->compute_attention(
                $query, $key, $value,
                $query_mask,
                $value_mask,
                $key_mask,
                $attention_mask,
                $useCausalMask,
                $training,
        );
        $container->attention_output = $attention_output;

        //
        // output_dense:   (B, (Tq), H, Dv) -> (B, (Tq), Fq)
        //
        [$full_input_shape,$full_output_shape,$dmy] = $this->make_forward_dense_shape(
            $attention_output, $Fq,
        );
        $attention_output = $attention_output->reshape($full_input_shape);
        $attention_output = $this->output_dense->_rawCall([$attention_output],['training'=>$training])[0];
        /** @var NDArray $attention_output */
        $attention_output = $attention_output->reshape($full_output_shape);


        //
        // keep values for differentiation
        //
        $container->attention_mask = $attention_mask;
        $container->training = $training;
        $container->scaled_query = $scaled_query;
        $container->key = $key;
        $container->value = $value;
        $container->attention_scores = $attention_scores;

        if($returnAttentionScores) {
            return [$attention_output, $attention_scores];
        }
        return $attention_output;
    }
    
    /**
     * @return array<NDArray>
     */
    protected function differentiate(NDArray $dOutputs) : array
    {
        $K = $this->backend;
        $container = $this->container();

        //
        // differential output dense
        //
        // dOutput:  (B,(Tq),Fq) => (B,(Tq),H,Dv)
        //
        [$full_output_shape,$full_input_shape] = $this->make_backward_dense_shape(
            $dOutputs,[$this->numHeads, $this->valueDim],
        );
        $dOutputs = $dOutputs->reshape($full_output_shape);
        $dAttention_output = $this->output_dense->_rawDifferentiate([$dOutputs])[0];
        $dAttention_output = $dAttention_output->reshape($full_input_shape);

        //
        // differential attention scores
        // 
        // dOutput: (B,(Tq),H,Dv)
        // dQuery:  (B,(Tq),H,Dk)
        // dKey:    (B,(Tv),H,Dk)
        // dValue:  (B,(Tv),H,Dv)
        [$dQuery, $dKey, $dValue] = $this->compute_differntiate_attention(
            $dAttention_output,
            $container->scaled_query,
            $container->key,
            $container->value,
            $container->attention_output,
            $container->attention_scores,
            $container->attention_mask,
            $container->training
        );


        //
        // differential value dense
        //
        // dValue: (B,(Tv),H,Dv) => (B,(Tv),Fv)
        //
        [$full_output_shape,$full_input_shape] = $this->make_backward_dense_shape(
            $dValue, $this->value_feature_shape,
        );
        $dValue = $dValue->reshape($full_output_shape);
        $dValue = $this->value_dense->_rawDifferentiate([$dValue])[0];
        $dValue = $dValue->reshape($full_input_shape);

        //
        // differential query dense
        //
        // dQuery: (B,(Tq),H,Dk) => (B,(Tq),Fq)
        //
        [$full_output_shape,$full_input_shape] = $this->make_backward_dense_shape(
            $dQuery, $this->query_feature_shape,
        );
        $dQuery = $dQuery->reshape($full_output_shape);
        $dQuery = $this->query_dense->_rawDifferentiate([$dQuery])[0];
        $dQuery = $dQuery->reshape($full_input_shape);

        //
        // differential key dense
        //
        // dKey:  (B,(Tv),H,Dk) => (B,(Tv),Fk)
        //
        [$full_output_shape,$full_input_shape] = $this->make_backward_dense_shape(
            $dKey, $this->key_feature_shape,
        );
        $dKey = $dKey->reshape($full_output_shape);
        $dKey   = $this->key_dense->_rawDifferentiate([$dKey])[0];
        $dKey = $dKey->reshape($full_input_shape);


        if($container->sameKey) {
            $K->update_add($dValue,$dKey);
            return [$dQuery,$dValue];
        } else {
            return [$dQuery,$dValue,$dKey];
        }
    }

    private function alloc_attention_mask(
        NDArray $query,
        NDArray $value,
    ) : NDArray
    {
        $K = $this->backend;
        $n_attn_axes = count($this->attention_axes);
        //$q_seq_length = $query->shape()[1];
        //$v_seq_length = ($value===null) ? $q_seq_length : $value->shape()[1];
        $q_seq_shape = array_slice($query->shape(),1,$n_attn_axes);
        $v_seq_shape = array_slice($value->shape(),1,$n_attn_axes);
        $batches = $query->shape()[0];
        $shape = array_merge([$batches],$q_seq_shape,$v_seq_shape);
        $attention_mask = $K->ones($shape,dtype:NDArray::bool);
        return $attention_mask;
    }
    
    private function compute_masked_attention_scores(
        NDArray $attention_scores,
        NDArray $query,
        NDArray $value,
        ?NDArray $query_mask=null,
        ?NDArray $value_mask=null,
        ?NDArray $key_mask=null,
        ?NDArray $attention_mask=null,
        ?bool $useCausalMask=null,
    ) : ?NDArray
    {
        $K = $this->backend;
        $m_inf = $this->mask_exp;

        if($value_mask && $key_mask) {
            if(spl_object_id($value_mask)==spl_object_id($key_mask)) {
                $key_mask = null;
            }
        }

        $auto_mask = null;
        if($useCausalMask) {
            //
            // causal mask:  ((Tq), (Tv))
            //
            $causal_mask = $this->compute_causal_mask($query, $value);
            if($query_mask==null && $value_mask==null && $key_mask==null) {
                // When causal mask is used alone, apply mask directly to save memory.
                // ((Tq), (Tv)) -> (1, 1, (Tq), (Tv)) -mask-to-> (B, H, (Tq), (Tv))
                $attention_scores = $K->masking($causal_mask,$attention_scores,fill:$m_inf,mode:1,axis:-$causal_mask->ndim());//mode=add
                return $attention_scores;
            }
            // If the causal mask does not exist alone, it is merged with other masks.
                // (B, (Tq), (Tv))
            $auto_mask = $this->alloc_attention_mask($query,$value);
            // (B, (Tq), (Tv)) <==masking== ((Tq), (Tv))
            $K->update_masking($auto_mask,$causal_mask,axis:-$causal_mask->ndim());
        }
        if($query_mask) {
            //echo $K->localMatrixOperator()->toString($query_mask,format:'%13.8e',indent:true)."\n";
            if($auto_mask==null && $value_mask==null && $key_mask==null) {
                // When query mask is used alone, apply mask directly to save memory.
                // (B, (Tq)) -> (B, 1, (Tq), 1) -mask-to-> (B, H, (Tq), (Tv))
                //echo $K->localMatrixOperator()->toString($attention_scores,format:'%13.8e',indent:true)."\n";
                $attention_scores = $K->masking($query_mask,$attention_scores,fill:$m_inf,mode:1,batchDims:1,axis:2);
                //echo $K->localMatrixOperator()->toString($attention_scores,format:'%13.8e',indent:true)."\n";
                return $attention_scores;
            }
            // If the query mask does not exist alone, it is merged with other masks.
            if($auto_mask==null) {
                // (B, (Tq), (Tv))
                $auto_mask = $this->alloc_attention_mask($query,$value);
            }
            // (B, (Tq), (Tv)) <==masking== (B, (Tq), 1)
            $K->update_masking($auto_mask,$query_mask);
        }
        if($value_mask) {
            if($auto_mask==null && $key_mask==null) {
                // When value mask is used alone, apply mask directly to save memory.
                // (B, (Tv)) -> (B, 1, 1, (Tv)) -mask-to-> (B, H, (Tq), (Tv))
                $attention_scores = $K->masking($value_mask,$attention_scores,fill:$m_inf,mode:1,batchDims:1,axis:-$value_mask->ndim()+1);
                return $attention_scores;
            }
            // If the value mask does not exist alone, it is merged with other masks.
            if($auto_mask==null) {
                // (B, (Tq), (Tv))
                $auto_mask = $this->alloc_attention_mask($query,$value);
            }
            // (B, (Tq), (Tv)) <==masking== (B, 1, (Tv))
            $K->update_masking($auto_mask,$value_mask,batchDims:1,axis:-$value_mask->ndim()+1);
        }
        if($key_mask) {
            if($auto_mask==null) {
                // When key mask is used alone, apply mask directly to save memory.
                // (B, (Tv)) -> (B, 1, 1, (Tv)) -mask-to-> (B, H, (Tq), (Tv))
                $attention_scores = $K->masking($key_mask,$attention_scores,fill:$m_inf,mode:1,batchDims:1,axis:-$key_mask->ndim()+1);
                return $attention_scores;
            }
            // If the key mask does not exist alone, it is merged with other masks.
            if($auto_mask==null) {
                // (B, (Tq), (Tv))
                $auto_mask = $this->alloc_attention_mask($query,$value);
            }
            // (B, (Tq), (Tv)) <==masking== (B, 1, (Tv))
            $K->update_masking($auto_mask,$key_mask,batchDims:1,axis:-$key_mask->ndim()+1);
        }

        // If it has any mask, it will be applied to the scores.
        if($auto_mask) {
            // (B,H,(Tq),(Tv)) <==masking== (B,(Tq),(Tv))
            $attention_scores = $K->masking($auto_mask,$attention_scores,fill:$m_inf,mode:1,batchDims:1,axis:-$auto_mask->ndim()+1);
        }
        return $attention_scores;
    }

    /**
     * causal mask:  ((Tq), (Tv))
     * 
     * This results in a triangular matrix like this.
     * [[True, False, False],
     *  [True, True,  False],
     *  [True, True,  True ] ]
     * 
     */
    private function compute_causal_mask(
        NDArray $query,
        ?NDArray $value=null
    ) : NDArray
    {
        $K = $this->backend;
        $n_attn_axes = count($this->attention_axes);
        $q_seq_shape = array_slice($query->shape(),1,$n_attn_axes);
        $v_seq_shape = ($value===null) ? $q_seq_shape : array_slice($value->shape(),1,$n_attn_axes);
        $q_seq_length = array_product($q_seq_shape);
        $v_seq_length = array_product($v_seq_shape);
        $ones_mask = $K->ones([$q_seq_length, $v_seq_length],dtype:NDArray::bool);
        $causal_mask = $K->bandpart($ones_mask,-1,0);
        $causal_mask = $causal_mask->reshape(array_merge($q_seq_shape,$v_seq_shape));

        // causal_mask: ((Tq), (Tv))
        return $causal_mask;
    }
    
    /**
     * @param array<int> $inputShape
     * @param array<int> $effectorDims
     * @return array{int,array<int>,array<int>,int,array<int>}
     */
    private function build_dense_args(
        array $inputShape,
        array $effectorDims,
    ) : array
    {
        if($this->attention_axes==null) {
            $this->attention_axes = range(1, count($inputShape)-2);
        }

        // input:   ((Batch, (T)), (Feature))
        // kernel:  ((Feature), （Heads)
        // output:  ((Batch, (T)), (Heads))

        // inputShape = (batch,(T),dim)
        // units = numHeads*headDim
        // dense_input_shape = (T,dim)
        $feature = $inputShape;
        $batch = array_shift($feature);
        $T = array_splice($feature,0,count($this->attention_axes));
        $units = array_product($effectorDims);
        $dense_input_shape = [array_product($T),array_product($feature)];
        return [$batch,$T,$feature,$units,$dense_input_shape];
    }

    /**
     * @param array<int> $effectorDims
     * @return array{array<int>,array<int>,array<int>}
     */
    private function make_forward_dense_shape(
        NDArray $inputs,
        array $effectorDims,
    ) : array
    {
        // input:   ((Batch, (T)), (Feature))
        // kernel:  ((Feature), （Heads)
        // output:  ((Batch, (T)), (Heads))

        $feature = $inputs->shape();
        $batch = array_shift($feature);
        $T = array_splice($feature,0,count($this->attention_axes));
        $dense_input_shape = [$batch*array_product($T),array_product($feature)];
        $full_output_shape = array_merge([$batch],$T,$effectorDims);
        return [$dense_input_shape,$full_output_shape,$feature];
    }

    /**
     * @param array<int> $feature
     * @return array{array<int>,array<int>}
     */
    private function make_backward_dense_shape(
        NDArray $dOutputs,
        array $feature,
    ) : array
    {
        $effectorDims = $dOutputs->shape();
        $batch = array_shift($effectorDims);
        $T = array_splice($effectorDims,0,count($this->attention_axes));
        //echo "T:".implode(',',$T)."\n";
        //echo "effectorDims:".implode(',',$effectorDims)."\n";
        $full_output_shape = [$batch*array_product($T),array_product($effectorDims)];
        $full_input_shape = array_merge([$batch],$T,$feature);
        return [$full_output_shape,$full_input_shape];
    }


    /**
     *  key(B.Tv.H.Dk),query(B.Tq.H.Dk)->scores(B.H.Tq.Tv)
     *  $this->dot_product_equation = 'a{e}cd,a{b}cd->ac{b}{e}';
     * 
     *  @return array{array<int>,array<int>,array<int>}
     */
    private function make_forward_product_shape(
        NDArray $key,
        NDArray $query,
    ) : array
    {
        $key_hd = $key->shape();
        $batch = array_shift($key_hd);
        $Tv = array_splice($key_hd,0,count($this->attention_axes));
        $key_input_shape = array_merge([$batch,array_product($Tv)],$key_hd);

        $query_hd = $query->shape();
        $batch = array_shift($query_hd);
        $Tq = array_splice($query_hd,0,count($this->attention_axes));
        $query_input_shape = array_merge([$batch,array_product($Tq)],$query_hd);

        $score_output_shape = array_merge([$batch,$query_hd[0]],$Tq,$Tv);
        return [$key_input_shape,$query_input_shape,$score_output_shape];
    }

    /**
     *  key(B.Tv.H.Dk),query(B.Tq.H.Dk)->scores(B.H.Tq.Tv)
     *  $this->backward_dot_product_key_equation   = 'ac{b}{e},a{b}cd->a{e}cd';
     *  $this->backward_dot_product_query_equation = 'ac{b}{e},a{e}cd->a{b}cd';
     * 
     *  @return array{array<int>,array<int>,array<int>}
     */
    private function make_backward_product_shape(
        NDArray $key,
        NDArray $query,
    ) : array
    {
        $key_hd = $key->shape();
        $batch = array_shift($key_hd);
        $Tv = array_splice($key_hd,0,count($this->attention_axes));
        $key_input_shape = array_merge([$batch,array_product($Tv)],$key_hd);

        $query_hd = $query->shape();
        $batch = array_shift($query_hd);
        $Tq = array_splice($query_hd,0,count($this->attention_axes));
        $query_input_shape = array_merge([$batch,array_product($Tq)],$query_hd);

        $score_output_shape = [$batch,$query_hd[0],array_product($Tq),array_product($Tv)];
        return [$key_input_shape,$query_input_shape,$score_output_shape];
    }

    /**
     *  scores: ((Batch), numHeads, (Tq), (Tv))
     *  value:  ((Batch), (Tv), numHeads, valueDim)
     *  output: ((Batch), (Tq), numHeads, valueDim)
     *  
     *  $this->combine_equation = 'ac{b}{e},a{e}cd->a{b}cd';
     * 
     *  @return array{array<int>,array<int>,array<int>}
     */
    private function make_forward_combine_shape(
        NDArray $scores,
        NDArray $value,
    ) : array
    {
        $scores_tv = $scores->shape();
        $batch = array_shift($scores_tv);
        $scores_nhd = array_shift($scores_tv);
        $Tq = array_splice($scores_tv,0,count($this->attention_axes));
        $scores_input_shape = [$batch,$scores_nhd,array_product($Tq),array_product($scores_tv)];

        $value_hd = $value->shape();
        $batch = array_shift($value_hd);
        $Tv = array_splice($value_hd,0,count($this->attention_axes));
        $value_input_shape = array_merge([$batch,array_product($Tv)],$value_hd);

        $combine_output_shape = array_merge([$batch],$Tq,$value_hd);
        return [$scores_input_shape,$value_input_shape,$combine_output_shape];
    }

    /**
     *   doutput: ((Batch), (Tq), numHeads, valueDim)
     *   dscores: ((Batch), numHeads, (Tq), (Tv))
     *   dvalue:  ((Batch), (Tv), numHeads, valueDim)
     *   
     *   $this->backward_combine_scores_equation = 'a{b}cd,a{e}cd->ac{b}{e}';
     *   $this->backward_combine_value_equation  = 'a{b}cd,ac{b}{e}->a{e}cd';
     * 
     *  @return array{array<int>,array<int>,array<int>}
     */
    private function make_backward_combine_shape(
        NDArray $scores,
        NDArray $value,
    ) : array
    {
        $scores_tv = $scores->shape();
        $batch = array_shift($scores_tv);
        $scores_nhd = array_shift($scores_tv);
        $Tq = array_splice($scores_tv,0,count($this->attention_axes));
        $scores_input_shape = [$batch,$scores_nhd,array_product($Tq),array_product($scores_tv)];

        $value_hd = $value->shape();
        $batch = array_shift($value_hd);
        $Tv = array_splice($value_hd,0,count($this->attention_axes));
        $value_input_shape = array_merge([$batch,array_product($Tv)],$value_hd);

        $combine_output_shape = array_merge([$batch,array_product($Tq)],$value_hd);
        return [$scores_input_shape,$value_input_shape,$combine_output_shape];
    }

}
