<?php
namespace Rindow\NeuralNetworks\Dataset;

use LogicException;
use Rindow\Math\Matrix\MatrixOperator;
use Interop\Polite\Math\Matrix\NDArray;

class Mnist
{
    protected $mo;
    protected $urlBase = 'http://yann.lecun.com/exdb/mnist/';
    protected $keyFiles = [
        'train_images'=>'train-images-idx3-ubyte.gz',
        'train_labels'=>'train-labels-idx1-ubyte.gz',
        'test_images'=>'t10k-images-idx3-ubyte.gz',
        'test_labels'=>'t10k-labels-idx1-ubyte.gz'
    ];
    protected $trainNum = 60000;
    protected $testNum = 10000;
    protected $imageShape = [1, 28, 28]; // = 784
    protected $datasetDir;
    protected $saveFile;

    public function __construct($mo)
    {
        $this->matrixOperator = $mo;
        $this->datasetDir = $this->getDatasetDir();
        if(!file_exists($this->datasetDir)) {
            @mkdir($this->datasetDir,0777,true);
        }
        $this->saveFile = $this->datasetDir . "/mnist.pkl";
    }
    protected function getDatasetDir()
    {
        return sys_get_temp_dir().'/rindow/nn/datasets/mnist';
    }

    protected function console($message)
    {
        fwrite(STDERR,$message);
    }

    public function loadData(string $filePath=null)
    {
        $mo = $this->matrixOperator;
        if($filePath===null) {
            $filePath = $this->saveFile;
        }
        if(file_exists($filePath)) {
            $dataset = $this->loadPickle($filePath);
        } else {
            $dataset = $this->getFiles($filePath);
        }

        return [[$dataset['train_images'], $dataset['train_labels']],
                [$dataset['test_images'],  $dataset['test_labels']]];
    }

    public function cleanPickle(string $filePath=null)
    {
        if($filePath===null) {
            $filePath = $this->saveFile;
        }
        unlink($this->saveFile);
    }

    protected function loadPickle($filePath)
    {
        $this->console("Loading pickle file ...");
        $data = file_get_contents($filePath);
        if(!$data)
            throw new LogicException('read error: '.$this->saveFile);
        $dataset = unserialize($data);
        unset($data);
        $this->console("Done!\n");
        return $dataset;
    }

    protected function getFiles($filePath)
    {
        $this->downloadFiles();
        $dataset = $this->convertNDArray();
        $this->console("Creating pickle file ...");
        //with open(save_file, 'wb') as f:
        //    pickle.dump(dataset, f, -1)
        file_put_contents($filePath,serialize($dataset));
        $this->console("Done!\n");
        return $dataset;
    }

    public function downloadFiles()
    {
        foreach($this->keyFiles as $key => $filename) {
            $this->download($filename);
        }
    }

    protected function download($filename)
    {
        $filePath = $this->datasetDir . "/" . $filename;

        if(file_exists($filePath))
            return;

        $this->console("Downloading " . $filename . " ... ");
        copy($this->urlBase.$filename, $filePath);
        $this->console("Done\n");
    }

    protected function convertNDArray()
    {
        $dataset = [];
        $dataset['train_images'] = $this->convertImage($this->keyFiles['train_images']);
        $dataset['train_labels'] = $this->convertLabel($this->keyFiles['train_labels']);
        $dataset['test_images']  = $this->convertImage($this->keyFiles['test_images']);
        $dataset['test_labels']  = $this->convertLabel($this->keyFiles['test_labels']);

        return $dataset;
    }

    protected function convertLabel($filename)
    {
        $mo = $this->matrixOperator;
        $filePath = $this->datasetDir . "/" . $filename;

        $this->console("Converting ".$filename." to NDArray ...");

        $len = $this->calcZippedDataLength($filePath);
        $size = (int)($len-8);
        $labels = $mo->zeros([$size],NDArray::uint8);
        $this->loadZipFile($filePath,8,$labels);
        $this->console("Done\n");

        return $labels;
    }

    protected function convertImage($filename)
    {
        $mo = $this->matrixOperator;
        $filePath = $this->datasetDir."/".$filename;
        $imageSize = (int)array_product($this->imageShape);

        $this->console("Converting ".$filename." to NDArray ...");

        $len = $this->calcZippedDataLength($filePath);
        $size = (int)(($len-16)/$imageSize);
        $data = $mo->zeros(array_merge([$size],$this->imageShape) ,NDArray::float32);
        $this->loadZipFile($filePath,16,$data);

        $this->console("Done\n");
        return $data;
    }

    protected function calcZippedDataLength($filePath)
    {
        $len = 0;
        $zp = gzopen($filePath,'rb');
        if($zp==false)
            throw new LogicException('file not found: '.$filePath);

        while($buf=gzread($zp,80960)) {
            $len += strlen($buf);
        }
        gzclose($zp);
        return $len;
    }

    protected function loadZipFile(string $filePath,int $offset,NDArray $data)
    {
        $buffer = $data->buffer();
        $i=$j=0;
        $zp  = gzopen($filePath,'rb');
        $buf = gzread($zp,$offset);
        while($buf=gzread($zp,8096)) {
            $values = unpack("C*",$buf);
            foreach ($values as $value) {
                $buffer[$i] = $value;
                $i++;
            }
            $j++;
            if($j>=200) {
                $j=0;
                $this->console('.');
            }
        }
        gzclose($zp);
    }
}
