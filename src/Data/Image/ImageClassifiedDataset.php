<?php
namespace Rindow\NeuralNetworks\Data\Image;

use Rindow\NeuralNetworks\Data\Dataset\ClassifiedDirectoryDataset;
use Interop\Polite\Math\Matrix\NDArray;
use ArrayObject;
use function Rindow\Math\Matrix\R;

class ImageClassifiedDataset extends ClassifiedDirectoryDataset
{
    protected $verbose;
    protected $classnames = [];
    protected $height;
    protected $width;
    protected $channels;
    protected $fit;
    protected $maxClassId;
    protected $dtype;
    protected $dtypeClassId;

    public function __construct(
        object $mo,
        string $path,
        int $height=null,
        int $width=null,
        int $channels=null,
        bool $fit=null,
        int $dtype=null,
        int $dtype_class_id=null,
        array $classnames=null,
        int $verbose=null,
        ...$options
        )
    {
        parent::__construct($mo, $path, ...$options);
        // defaults
        $height = $height ?? 32;
        $width = $width ?? 32;
        $channels = $channels ?? 3;
        $fit = $fit ?? true;
        $dtype = $dtype ?? NDArray::uint8;
        $dtype_class_id = $dtype_class_id ?? NDArray::int32;
        $classnames = $classnames ?? [];
        $verbose = $verbose ?? null;

        $this->height = $height;
        $this->width = $width;
        $this->channels = $channels;
        $this->fit = $fit;
        $this->classnames = array_flip($classnames);
        $this->maxClassId = array_reduce($this->classnames,'max',-1);
        $this->dtype = $dtype;
        $this->dtypeClassId = $dtype_class_id;
        $this->verbose = $verbose;
    }

    protected function console($message)
    {
        if($this->verbose) {
            if(defined('STDERR')) {
                fwrite(STDERR,$message);
            }
        }
    }

    protected function readContents($filename)
    {
        $ext = strtolower(pathinfo($filename,PATHINFO_EXTENSION));
        switch($ext) {
            case 'bmp': {
                $image = imagecreatefrombmp($filename);
                break;
            }
            case 'gif': {
                $image = imagecreatefromgif($filename);
                break;
            }
            case 'jpg':
            case 'jpeg': {
                $image = imagecreatefromjpeg($filename);
                break;
            }
            case 'png': {
                $image = imagecreatefrompng($filename);
                break;
            }
            default: {
                break;
            }
        }
        $height = $this->height;
        $width = $this->width;
        $channels = 3;
        $array = $this->mo->zeros(
            [$height, $width, $channels], $this->dtype);
        $imHeight = imagesy($image);
        $imWidth  = imagesx($image);
        if($this->fit && ($height!=$imHeight||$width!=$imWidth)) {
            $aspectRatio = $width/$height;
            $imAspectRatio = $imWidth/$imHeight;
            if($imAspectRatio>$aspectRatio) {
                $offsety = (int)floor(($height-$width/$imAspectRatio)/2);
                $offsetx = 0;
                $copyWidth = $width;
                $copyHeight = $height-$offsety*2;
            } else {
                $offsetx = (int)floor(($width-$height*$imAspectRatio)/2);
                $offsety = 0;
                $copyWidth = $width-$offsetx*2;
                $copyHeight = $height;
            }
            $fitImage = imagecreatetruecolor($width, $height);
            imagecopyresampled($fitImage, $image,
                $offsetx, $offsety, 0, 0,
                $copyWidth, $copyHeight, $imWidth, $imHeight);
            $imHeight = $height;
            $imWidth = $width;
            imagedestroy($image);
            $image = $fitImage;
        }
        $offsety = (int)floor(($imHeight-$height)/2);
        $offsetx = (int)floor(($imWidth-$width)/2);
        for($y=0;$y<$height;$y++) {
            for($x=0;$x<$width;$x++) {
                $py = $y+$offsety;
                $px = $x+$offsetx;
                if($py<0 || $py>=$imHeight || $px<0 || $px>=$imWidth) {
                    continue;
                }
                $rgb = imagecolorat($image, $px, $py);
                for($c=0;$c<$channels;$c++) {
                    $array[$y][$x][$c] = ($rgb >> (($channels-$c-1)*8)) & 0xFF;
                }
            }
        }
        imagedestroy($image);
        return $array;
    }

    protected function makeBatchInputs($inputs)
    {
        $channels = 3;
        $batchSize = count($inputs);
        $batch = $this->mo->zeros(
            [$batchSize, $this->height, $this->width, $channels], $this->dtype);
        $la = $this->mo->la();
        foreach ($inputs as $rowid => $content) {
            $la->copy($content,$batch[$rowid]);
        }
        return $batch;
    }

    protected function makeBatchTests($tests)
    {
        $batchSize = count($tests);
        $batch = $this->mo->zeros([$batchSize], $this->dtypeClassId);
        foreach ($tests as $rowid => $classname) {
            if(array_key_exists($classname, $this->classnames)) {
                $classId = $this->classnames[$classname];
            } else {
                $this->maxClassId++;
                $classId = $this->maxClassId;
                $this->classnames[$classname] = $classId;
            }
            $batch[$rowid] = $classId;
        }
        return $batch;
    }

    public function classnames()
    {
        return array_flip($this->classnames);
    }

    public function loadData()
    {
        $la = $this->mo->la();
        $this->console("Generating filename list ...");
        $filenames = $this->getFilenames();
        $totalSize = count($filenames);
        $this->console(" Done. Total=$totalSize\n");
        $inputs = $la->alloc(
            [$totalSize,$this->height,$this->width,$this->channels],
            $this->dtype);
        $tests = $la->alloc([$totalSize],$this->dtypeClassId);
        $this->console("Loading ...\n");
        $dataset = $this->getIterator();
        $nn=0;
        $startTime = time();
        foreach ($dataset as $key => $value) {
            [$batchInputs,$batchTests] = $value;
            $idx = $key*$this->batchSize;
            $batchSize = count($batchInputs);
            $la->copy($batchInputs,$inputs[R($idx,$idx+$batchSize)]);
            $la->copy($batchTests,$tests[R($idx,$idx+$batchSize)]);
            $nn++;
            if($nn>10) {
                $nn=0;
                $this->progressBar($idx+$batchSize,$totalSize,$startTime,25);
            }
        }
        $this->console("\nDone.\n");
        return [$inputs,$tests];
    }
}
