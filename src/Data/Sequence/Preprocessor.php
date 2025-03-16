<?php
namespace Rindow\NeuralNetworks\Data\Sequence;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

class Preprocessor
{
    protected object $mo;

    public function __construct(object $mo)
    {
        $this->mo = $mo;
    }

    /**
     * @param iterable<iterable<int>> $sequences
     */
    public function padSequences(
        iterable $sequences,
        ?int $maxlen=null,
        ?int $dtype=null,
        ?string $padding=null,
        ?string $truncating=null,
        float|int|bool|null $value=null,
    ) : NDArray
    {
        // defaults
        $maxlen = $maxlen ?? null;
        $dtype = $dtype ?? NDArray::int32;
        $padding = $padding ?? "pre";
        $truncating = $truncating ?? "pre";
        $value = $value ?? 0;
        
        $max = 0;
        $size = 0;
        foreach ($sequences as $sequence) {
            if(!is_iterable($sequence)) {
                throw new InvalidArgumentException('sequences must be list of iterable');
            }
            $len = count($sequence);
            $max = max($len,$max);
            $size++;
        }
        if($maxlen!=null) {
            if($max >= $maxlen) {
                $max = $maxlen;
            }
        }
        if($dtype==NDArray::float32||$dtype==NDArray::float64) {
            $value = (float)$value;
        }
        $tensor = $this->mo->full([$size,$max],$value,$dtype);
        $i = 0;
        foreach ($sequences as $sequence) {
            $j = 0;
            $len = count($sequence);
            $skip = 0;
            if($len>$max) {
                if($truncating=='pre') {
                    $skip = $len-$max;
                } elseif($truncating=='post') {
                    ;
                } else {
                    throw new InvalidArgumentException('trancating must be "pre" or "post".');
                }
            }
            if($len<$max) {
                if($padding=='pre') {
                    $j = $max-$len;
                } elseif($padding=='post') {
                    ;
                } else {
                    throw new InvalidArgumentException('padding must be "pre" or "post".');
                }
            }
            foreach ($sequence as $val) {
                if($skip>0) {
                    $skip--;
                    continue;
                }
                if($j>=$max)
                    break;
                $tensor[$i][$j] = $val;
                $j++;
            }
            $i++;
        }
        return $tensor;
    }
}
