<?php
namespace Rindow\NeuralNetworks\Dataset;

use LogicException;
use RuntimeException;
use Rindow\Math\Matrix\MatrixOperator;
use Interop\Polite\Math\Matrix\NDArray;

class Mnist
{
    protected object $matrixOperator;
    // Unable to download from original site.
    // http://yann.lecun.com/exdb/mnist/'
    // Redirecting download to alternative mirror site.
    protected string $urlBase = 'https://storage.googleapis.com/cvdf-datasets/mnist/';
    /** @var array<string,string> $keyFiles */
    protected $keyFiles = [
        'train_images'=>'train-images-idx3-ubyte.gz',
        'train_labels'=>'train-labels-idx1-ubyte.gz',
        'test_images'=>'t10k-images-idx3-ubyte.gz',
        'test_labels'=>'t10k-labels-idx1-ubyte.gz'
    ];
    protected int $trainNum = 60000;
    protected int $testNum = 10000;
    /** @var array<int> $imageShape */
    protected array $imageShape = [1, 28, 28]; // = 784
    protected string $datasetDir;
    protected string $saveFile;

    public function __construct(object $mo)
    {
        $this->matrixOperator = $mo;
        $this->datasetDir = $this->getDatasetDir();
        if(!file_exists($this->datasetDir)) {
            @mkdir($this->datasetDir,0777,true);
        }
        $this->saveFile = $this->datasetDir . "/mnist.pkl";
    }

    public function datasetDir() : string
    {
        return $this->datasetDir;
    }

    protected function getRindowDatesetDir() : string
    {
        $dataDir = getenv('RINDOW_NEURALNETWORKS_DATASETS');
        if(!$dataDir) {
            $dataDir = sys_get_temp_dir().'/rindow/nn/datasets';
        }
        return $dataDir;
    }

    protected function getDatasetDir() : string
    {
        return $this->getRindowDatesetDir().'/mnist';
    }

    protected function console(string $message) : void
    {
        fwrite(STDERR,$message);
    }

    /**
     * @return array{array{NDArray,NDArray},array{NDArray,NDArray}}
     */
    public function loadData(?string $filePath=null) : array
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

    public function cleanPickle(?string $filePath=null) : void
    {
        if($filePath===null) {
            $filePath = $this->saveFile;
        }
        unlink($this->saveFile);
    }

    /**
     * @return array<string,NDArray>
     */
    protected function loadPickle(string $filePath) : array
    {
        $this->console("Loading pickle file ...");
        $data = file_get_contents($filePath);
        if(!$data)
            throw new LogicException('read error: '.$this->saveFile);
        $dataset = $this->matrixOperator->unserializeArray($data);
        unset($data);
        $this->console("Done!\n");
        return $dataset;
    }

    /**
     * @return array<string,NDArray>
     */
    protected function getFiles(string $filePath) : array
    {
        $this->downloadFiles();
        $dataset = $this->convertNDArray();
        $this->console("Creating pickle file ...");
        //with open(save_file, 'wb') as f:
        //    pickle.dump(dataset, f, -1)
        file_put_contents($filePath,$this->matrixOperator->serializeArray($dataset));
        $this->console("Done!\n");
        return $dataset;
    }

    public function downloadFiles() : void
    {
        foreach($this->keyFiles as $key => $filename) {
            $this->download($filename);
        }
    }

    protected function download(string $filename) : void
    {
        $filePath = $this->datasetDir . "/" . $filename;

        if(file_exists($filePath))
            return;
        $this->console("Downloading " . $filename . " ... ");
        copy($this->urlBase.$filename, $filePath);
        $this->console("Done\n");
    }

    /**
     * @return array<string,NDArray>
     */
    protected function convertNDArray() : array
    {
        $dataset = [];
        $dataset['train_images'] = $this->convertImage($this->keyFiles['train_images']);
        $dataset['train_labels'] = $this->convertLabel($this->keyFiles['train_labels']);
        $dataset['test_images']  = $this->convertImage($this->keyFiles['test_images']);
        $dataset['test_labels']  = $this->convertLabel($this->keyFiles['test_labels']);

        return $dataset;
    }

    protected function convertLabel(string $filename) : NDArray
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

    protected function convertImage(string $filename) : NDArray
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

    protected function calcZippedDataLength(string $filePath) : int
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

    protected function loadZipFile(string $filePath,int $offset,NDArray $data) : void
    {
        $buffer = $data->buffer();
        $bfsz = count($buffer);
        $i=$j=0;
        $zp  = gzopen($filePath,'rb');
        $buf = gzread($zp,$offset);
        while($buf=gzread($zp,8096)) {
            $values = unpack("C*",$buf);
            foreach ($values as $value) {
                if($i<$bfsz) {
                    $buffer[$i] = $value;
                }
                $i++;
            }
            $j++;
            if($j>=200) {
                $j=0;
                $this->console('.');
            }
        }
        gzclose($zp);
        if($i!=$bfsz) {
            throw new RuntimeException("File ".$filePath." is probably broken. Please remove and reload.");
        }
    }
}
