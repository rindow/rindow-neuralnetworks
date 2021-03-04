<?php
namespace Rindow\NeuralNetworks\Data\Sequence;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;
use InvalidArgumentException;

class Preprocessor
{
    use GenericUtils;
    protected $mo;

    public function __construct($mo)
    {
        $this->mo = $mo;
    }

    public function padSequences(iterable $sequences, array $options=null) : NDArray
    {
        extract($this->extractArgs([
            'maxlen'=>null,
            'dtype'=>NDArray::int32,
            'padding'=>"pre",
            'truncating'=>"pre",
            'value'=>0,
        ],$options));
        if(!is_iterable($sequences)) {
            throw new InvalidArgumentException('sequences must be iterable.');
        }
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
