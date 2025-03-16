<?php
namespace Rindow\NeuralNetworks\Layer;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use Rindow\NeuralNetworks\Gradient\Variable;

/**
 *
 */
abstract class AbstractImage extends AbstractLayer
{
    protected int $rank;
    protected ?string $data_format;

    protected function normalizeInputShape(array|Variable|null $variable=null) : array
    {
        $inputShape=parent::normalizeInputShape($variable);
        if($this->rank==0) {
            return $inputShape;
        }
        if(count($inputShape)!=$this->rank+1) {
            throw new InvalidArgumentException(
                'Unsuppored input shape: ['.implode(',',$inputShape).']');
        }
        return $inputShape;
    }

    protected function getChannels() : int
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

    /**
     * @param null|int|array<int> $size
     * @param int|array<int> $default
     * @return array<int>
     */
    protected function normalizeFilterSize(
        null|int|array $size,
        string $sizename,
        int|array|null $default=null,
        ?bool $notNull=null) : array
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
