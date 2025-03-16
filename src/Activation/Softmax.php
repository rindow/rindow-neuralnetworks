<?php
namespace Rindow\NeuralNetworks\Activation;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

class Softmax extends AbstractActivation
{
    protected function call(NDArray $inputs, ?bool $training=null, ?NDArray $mask=null) : NDArray
    {
        $K = $this->backend;
        if($mask===null) {
            //echo "No mask\n";
            //echo "inputs=".$K->localMatrixOperator()->shapeToString($inputs->shape())."\n";
            //echo $K->localMatrixOperator()->toString($inputs,indent:true)."\n";
            //
            // Yk = exp(Ak + C') / sum(exp(Ai + C'))
            //
            $outputs = $K->softmax($inputs);
            $this->states->outputs = $outputs;
            //echo "outputs=".$K->localMatrixOperator()->shapeToString($outputs->shape())."\n";
            return $outputs;
        }
        //
        // masked softmax
        //
        // inputs = (batchs, seq, size)
        // mask   = (batchs, seq)
        //
        $ndim = $mask->ndim();
        $orignalInputShape = $inputs->shape();
        $batchShape = $orignalInputShape;
        $inputShape = array_splice($batchShape,-$ndim);
        //echo "inputs=".$K->localMatrixOperator()->shapeToString($inputs->shape())."\n";
        //echo "mask=".$K->localMatrixOperator()->shapeToString($mask->shape())."\n";
        //echo "ndim=$ndim\n";
        //echo "batchShape=".$K->localMatrixOperator()->shapeToString($batchShape)."\n";
        //echo "inputShape=".$K->localMatrixOperator()->shapeToString($inputShape)."\n";
        if($inputShape!=$mask->shape()) {
            throw new InvalidArgumentException('unmatch shape of inputs and mask: '.
                'inputs=('.implode(',',$inputs->shape()).'), '.
                'mask=('.implode(',',$mask->shape()).')'
            );
        }

        //echo $K->localMatrixOperator()->toString($mask,indent:true)."\n";
        //echo "mask=".$K->localMatrixOperator()->shapeToString($mask->shape())."\n";
        //$outputs = $K->softmax($inputs);
        //$nums = $K->sum($mask,axis:1);
        //echo "nums=".$K->localMatrixOperator()->shapeToString($nums->shape())."\n";
        //echo $K->localMatrixOperator()->toString($nums,indent:true)."\n";
        $inputs = $K->mul($inputs,$mask);
        //echo "masked_inputs=".$K->localMatrixOperator()->shapeToString($inputs->shape())."\n";
        //echo $K->localMatrixOperator()->ToString($inputs,indent:true)."\n";
        $batches = (int)array_product($batchShape);

        //mask = tf.cast(mask, tf.float32)
        //masked_inputs = inputs * mask
        //exp_x = tf.exp(masked_inputs - tf.reduce_max(masked_inputs, axis=-1, keepdims=True)) * mask
        //return exp_x / tf.reduce_sum(exp_x, axis=-1, keepdims=True)

        ////////////////
        //
        // C(m) = reduce_max(m,n)
        // Y(m,n) = exp(A(m,n) + C') / sum(exp(Ai + C'))
        //
        // inputs = inputs * mask
        // max = reduce_max(inputs)
        // exp_diff = exp(inputs-max)
        // sum_exp = sum( exp_diff*mask )
        // softmax = exp_diff / sum_exp
        /////////////////
        if(count($inputShape)==1) {
            array_unshift($inputShape,1);
        }
        if(count($inputShape)!=2) {
            throw new InvalidArgumentException('inputs must have 2d details: '.
                'inputs=('.implode(',',$inputs->shape()).'), '.
                'mask=('.implode(',',$mask->shape()).')'
            );
        }
        [$rows,$cols] = $inputShape;
        $inputs = $inputs->reshape([$batches*$rows,$cols]);
        //echo "flat_inputs=".$K->localMatrixOperator()->shapeToString($inputs->shape())."\n";
        //
        // expDiff = exp(inputs-max(inputs,axis=-1))
        //
        $maxes = $K->max($inputs,axis:-1);
        //echo "maxes=".$K->localMatrixOperator()->shapeToString($maxes->shape())."\n";
        //echo $K->localMatrixOperator()->ToString($maxes,indent:true)."\n";
        $expDiff = $K->exp($K->sub($inputs,$maxes,trans:true));
        $expDiff = $expDiff->reshape([$batches,$rows,$cols]);
        //echo "expDiff=".$K->localMatrixOperator()->shapeToString($maxes->shape())."\n";
        //echo $K->localMatrixOperator()->ToString($expDiff,'%3.3f',indent:true)."\n";
        //
        // expDiff *= mask
        //
        $expDiff = $K->mul($expDiff,$mask);
        //echo "musked_expDiff=".$K->localMatrixOperator()->shapeToString($maxes->shape())."\n";
        //echo $K->localMatrixOperator()->ToString($expDiff,'%3.3f',indent:true)."\n";
        //
        // sumExp = sum(expDiff, axis=-1)
        //
        $sumExp = $K->sum($expDiff,axis:-1);
        $expDiff = $expDiff->reshape([$batches*$rows,$cols]);
        $sumExp = $sumExp->reshape([$batches*$rows]);
        //echo "expDiff=".$K->localMatrixOperator()->shapeToString($maxes->shape())."\n";
        //echo $K->localMatrixOperator()->ToString($expDiff,'%3.3f',indent:true)."\n";
        //echo "sumExp=".$K->localMatrixOperator()->shapeToString($sumExp->shape())."\n";
        //echo $K->localMatrixOperator()->ToString($sumExp,'%3.3f',indent:true)."\n";
        //
        // outputs = expDiff / sumExp
        //
        $outputs = $K->div($expDiff, $sumExp, trans:true);
        //echo "outputs=".$K->localMatrixOperator()->shapeToString($outputs->shape())."\n";
        //echo $K->localMatrixOperator()->ToString($outputs,'%3.3f',indent:true)."\n";
        $outputs = $outputs->reshape($orignalInputShape);

        $this->states->outputs = $outputs;
        return $outputs;
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        //echo "dsoftmax:dOutputs=".$K->localMatrixOperator()->shapeToString($dOutputs->shape())."\n";
        //echo $K->localMatrixOperator()->toString($dOutputs,indent:true)."\n";
        //echo "outputs=".$K->localMatrixOperator()->toString($this->states->outputs,'%10.7e',indent:true)."\n";
        $dInputs = $K->dSoftmax($dOutputs, $this->states->outputs);
        //echo "dInputs=".$K->localMatrixOperator()->toString($dInputs,'%10.7e',indent:true)."\n";
        return $dInputs;
    }
}
