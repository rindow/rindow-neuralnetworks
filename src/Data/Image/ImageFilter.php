<?php
namespace Rindow\NeuralNetworks\Data\Image;

use Rindow\NeuralNetworks\Data\Dataset\DatasetFilter;
use Rindow\NeuralNetworks\Support\GenericUtils;
use InvalidArgumentException;

class ImageFilter implements DatasetFilter
{
    use GenericUtils;
    protected $mo;
    protected $inputs;
    protected $tests;
    protected $batchSize;
    protected $shuffle;
    protected $filter;
    protected $heightShiftInteger;
    protected $widthShiftInteger;

    public function __construct(
        $mo,
        array $options=null,
        array &$leftargs=null
        )
    {
        extract($this->extractArgs([
            'data_format'=>'channels_last',
            'height_shift'=>0,
            'width_shift'=>0,
            'vertical_flip'=>false,
            'horizontal_flip'=>false
        ],$options,$leftargs));
        $this->mo = $mo;
        if($data_format=='channels_last') {
            $channels_first = false;
        } elseif($data_format=='channels_first') {
            $channels_first = true;
        } else {
            throw new InvalidArgumentException('Unknown data format:'.$data_format);
        }
        $this->channelsFirst = $channels_first;
        if($height_shift<0) {
            throw new InvalidArgumentException('height_shift must be greater than 0 or equal');
        }
        $this->heightShift = $height_shift;
        if($width_shift<0) {
            throw new InvalidArgumentException('width_shift must be greater than 0 or equal');
        }
        $this->widthShift = $width_shift;
        $this->verticalFlip = $vertical_flip;
        $this->horizontalFlip = $horizontal_flip;
    }

    protected function adjustShiftSize($shape)
    {
        $batchSize = array_shift($shape);
        if($this->channelsFirst) {
            $channels = array_shift($shape);
        } else {
            $channels = array_pop($shape);
        }
        [$height,$width] = $shape;
        if($this->heightShift<1&&$this->heightShift!=0) {
            $heightShiftInteger = (int)floor($height*$this->heightShift);
        } else {
            $heightShiftInteger = (int)floor($this->heightShift);
        }
        $this->heightShiftInteger = $heightShiftInteger;
        $this->heightShiftLow = -$heightShiftInteger;
        $this->heightShiftHigh = $heightShiftInteger;
        if($this->widthShift<1&&$this->widthShift!=0) {
            $widthShiftInteger = (int)floor($width*$this->widthShift);
        } else {
            $widthShiftInteger = (int)floor($this->widthShift);
        }
        $this->widthShiftInteger = $widthShiftInteger;
        $this->widthShiftLow = -$widthShiftInteger;
        $this->widthShiftHigh = $widthShiftInteger;
    }

    public function translate($inputs, $tests=null) : array
    {
        $la = $this->mo->la();
        if($this->heightShiftInteger===null) {
            $this->adjustShiftSize($inputs->shape());
        }
        mt_srand();
        $newInputs = $la->alloc($inputs->shape(),$inputs->dtype());
        foreach($inputs as $key => $image) {
            if($this->heightShiftInteger) {
                $heightShiftInteger = mt_rand($this->heightShiftLow,$this->heightShiftHigh);
            } else {
                $heightShiftInteger = 0;
            }
            if($this->widthShiftInteger) {
                $widthShiftInteger = mt_rand($this->widthShiftLow,$this->widthShiftHigh);
            } else {
                $widthShiftInteger = 0;
            }
            if($this->verticalFlip) {
                $verticalFlip = mt_rand(0,1) ? true : false;
            } else {
                $verticalFlip = false;
            }
            if($this->horizontalFlip) {
                $horizontalFlip = mt_rand(0,1) ? true : false;
            } else {
                $horizontalFlip = false;
            }
            $la->imagecopy(
                $image,
                $newInputs[$key],
                $this->channelsFirst,
                $heightShiftInteger,
                $widthShiftInteger,
                $verticalFlip,
                $horizontalFlip
            );
        }
        return [$newInputs,$tests];
    }
}
