<?php
namespace Rindow\NeuralNetworks\Layer;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

/**
 *
 */
abstract class AbstractImage extends AbstractLayer
{
    protected function normalizeInputShape($variable=null) : array
    {
        $inputShape=parent::normalizeInputShape($variable);
        if($this->rank===null)
            return $inputShape;
        if(count($inputShape)!=$this->rank+1) {
            throw new InvalidArgumentException(
                'Unsuppored input shape: ['.implode(',',$inputShape).']');
        }
        return $inputShape;
    }

    protected function getChannels()
    {
        $inputShape = $this->inputShape;
        if($this->data_format==null||
           $this->data_format=='channels_last') {
            $channels = array_pop(
                $inputShape);
        } elseif($this->data_format=='channels_first') {
            $channels = array_unshift(
                $inputShape);
        } else {
            throw new InvalidArgumentException('data_format is invalid');
        }
        return $channels;
    }

    protected function normalizeFilterSize(
        $size,
        string $sizename,
        $default=null,
        $notNull=null)
    {
        if($size===null && !$notNull) {
            $size = $default;
        }
        if(is_int($size))
            return array_fill(0,$this->rank, $size);
        if(is_array($size)) {
            if(count($size)!=$this->rank) {
               throw new InvalidArgumentException("$sizename does not mach rank.");

            }
            return $size;
        }
        throw new InvalidArgumentException("$sizename must be array or integer.");
    }
}
