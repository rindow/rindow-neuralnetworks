<?php
namespace Rindow\NeuralNetworks\Dataset;

use LogicException;
use RuntimeException;
use Traversable;
use IteratorAggregate;
use Rindow\Math\Matrix\MatrixOperator;
use Interop\Polite\Math\Matrix\NDArray;
use PharData;
use function Rindow\Math\Matrix\R;

class Cifar10
{
    const DATA_BATCH_SIZE = 10000;
    const NUM_OF_DATA_FILE = 5;
    protected object $matrixOperator;
    protected string $baseUrl = 'https://www.cs.toronto.edu/~kriz/';
    protected string $downloadFile = 'cifar-10-binary.tar.gz';
    /** @var array<string,string> $keyFiles */
    protected $keyFiles = [
        'data_file_1'=>'data_batch_1.bin',
        'data_file_2'=>'data_batch_2.bin',
        'data_file_3'=>'data_batch_3.bin',
        'data_file_4'=>'data_batch_4.bin',
        'data_file_5'=>'data_batch_5.bin',
        'test_file'=>'test_batch.bin',
    ];
    protected int $trainNum = 50000;
    protected int $testNum = 10000;
    /** @var array<int> $imageShape */
    protected array $imageShape = [32, 32, 3]; // = 3072
    protected string $datasetDir;
    protected string $saveFile;

    public function __construct(object $mo)
    {
        $this->matrixOperator = $mo;
        $this->datasetDir = $this->getDatasetDir();
        if(!file_exists($this->datasetDir)) {
            @mkdir($this->datasetDir,0777,true);
        }
        $this->saveFile = $this->datasetDir . "/test_batch_labels.pkl";
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
        return $this->getRindowDatesetDir().'/cifar-10-batches-bin';
    }

    protected function console(string $message) : void
    {
        fwrite(STDERR,$message);
    }

    /**
     * @return array{Traversable<array{NDArray,NDArray}>,Traversable<array{NDArray,NDArray}>}
     */
    public function loadData(?string $filePath=null) : array
    {
        $mo = $this->matrixOperator;
        if($filePath===null) {
            $filePath = $this->saveFile;
        }
        if(!file_exists($filePath)) {
            $this->downloadFiles();
            $this->convertNDArray();
        }

        $dataset = $this->loadPickles();
        return $dataset;
    }

    public function cleanPickle(?string $filePath=null) : void
    {
        if($filePath===null) {
            $filePath = $this->saveFile;
        }
        $filenames = $this->keyFiles;
        foreach($filenames as $filename) {
            $filePath = $this->datasetDir.'/'.basename($filename,'.bin');
            unlink($filePath.'_images.pkl');
            unlink($filePath.'_labels.pkl');
        }
    }

    /**
     * @return array{Traversable<array{NDArray,NDArray}>,Traversable<array{NDArray,NDArray}>}
     */
    protected function loadPickles() : array
    {
        $filenames = $this->keyFiles;
        $testFiles = [array_pop($filenames)];

        $train = $this->createIterator($filenames);
        $test = $this->createIterator($testFiles);
        return [$train,$test];
    }

    /**
     * @param array<string> $filenames
     * @return Traversable<array{NDArray,NDArray}>
     */
    protected function createIterator($filenames) : iterable
    {
        $iter = new class (
            $this->matrixOperator, $filenames, $this->datasetDir
            ) implements IteratorAggregate
        {
            protected object $matrixOperator;
            /** @var array<string> $filenames */
            protected array $filenames;
            protected string $datasetDir;
            /**
             * @param array<string> $filenames
             */
            public function __construct(
                object $mo,
                array $filenames,
                string $datasetDir
                )
            {
                $this->matrixOperator = $mo;
                $this->filenames = $filenames;
                $this->datasetDir = $datasetDir;
            }

            public function getIterator(): Traversable
            {
                $filenames = $this->filenames;
                foreach($filenames as $filename) {
                    $filePath = $this->datasetDir."/".basename($filename,'.bin');
                    $data = file_get_contents($filePath.'_images.pkl');
                    if(!$data) {
                        throw new LogicException('read error: '.$filePath.'_images.pkl');
                    }
                    $images = $this->matrixOperator->unserializeArray($data);
                    unset($data);
                    $data = file_get_contents($filePath.'_labels.pkl');
                    if(!$data) {
                        throw new LogicException('read error: '.$filePath.'_labels.pkl');
                    }
                    $labels = $this->matrixOperator->unserializeArray($data);
                    unset($data);
                    yield [$images,$labels];
                }
            }
        };
        return $iter;
    }

    public function downloadFiles() : void
    {
        $this->download($this->downloadFile);
    }

    protected function download(string $filename) : void
    {
        $filePath = $this->datasetDir . "/" . $filename;

        if(!file_exists($filePath)){
            $this->console("Downloading " . $filename . " ... ");
            copy($this->baseUrl.$filename, $filePath);
            $this->console("Done\n");
        }

        if(file_exists($this->datasetDir.'/test_batch.bin')){
            return;
        }
        $this->console("Extract to:".$this->datasetDir.'/..'."\n");
        $files = [];
        foreach($this->keyFiles as $file){
            $files[]='cifar-10-batches-bin/'.$file;
        }
        $phar = new PharData($filePath);
        $rc=$phar->extractTo($this->datasetDir.'/..',$files,true);
        $this->console("Done\n");
    }

    protected function convertNDArray() : void
    {
        $mo = $this->matrixOperator;

        $filenames = $this->keyFiles;
        $testFiles = [array_pop($filenames)];

        $batches = self::DATA_BATCH_SIZE;
        $this->convertDataset(
            $filenames,
            $batches,
        );

        $batches = self::DATA_BATCH_SIZE;
        $this->convertDataset(
            $testFiles,
            $batches,
        );
    }

    /**
     * @param array<string> $filenames
     */
    protected function convertDataset(
        array $filenames,
        int $batches,
        ) : void
    {
        $mo = $this->matrixOperator;
        $images = $mo->zeros([$batches,32,32,3],NDArray::uint8);
        $labels = $mo->zeros([$batches],NDArray::uint8);
        foreach($filenames as $filename) {
            $pklname = basename($filename,'.bin');
            $this->convertImage(
                $filename, $pklname, $images, $labels
            );
        }
    }

    protected function convertImage(
        string $filename,
        string $pklname,
        NDArray $images,
        NDArray $labels
        ) : void
    {
        $mo = $this->matrixOperator;
        $filePath = $this->datasetDir."/".$filename;
        $labelPath = $this->datasetDir."/".$pklname."_labels.pkl";
        $imagePath = $this->datasetDir."/".$pklname."_images.pkl";
        $imageSize = array_product($this->imageShape);

        $this->console("Converting ".$filename." to NDArray ...");
        $p = 0;
        $f = fopen($filePath,'rb');
        if($f===false) {
            throw new RuntimeException('file not found:'.$filePath);
        }
        $j=0;
        while(true){
            $label = fread($f,1);
            if($label===false||$label=='') {
                break;
            }
            if($labels->size()<=$p){
                throw new RuntimeException('label buffer overflow');
            }
            $this->unpackLabel(
                $label,
                $labels[[$p,$p+1]]);
            $red = fread($f,1024);
            $green = fread($f,1024);
            $blue = fread($f,1024);
            if($red===false||
                $green===false||
                $blue===false) {
                break;
            }
            $this->unpackImage(
                $red,$green,$blue,
                $images[$p]);
            $p++;
            $j++;
            if($j>=200) {
                $j=0;
                $this->console('.');
            }
        }
        fclose($f);
        $this->console("\nCreating pickle file ...");
        file_put_contents($labelPath,$this->matrixOperator->serializeArray($labels));
        file_put_contents($imagePath,$this->matrixOperator->serializeArray($images));
        $this->console("Done\n");
    }

    protected function unpackLabel(
        string $data,
        NDArray $buffer
        ) : void
    {
        $values = unpack("C*",$data);
        $i=0;
        foreach ($values as $value) {
            $buffer[$i] = $value;
            $i++;
        }
    }
    
    protected function unpackImage(
        string $reddata,
        string $greendata,
        string $bluedata,
        NDArray $buffer
        ) : void
    {
        $buffer = $buffer->reshape([$buffer->size()]);
        $red = unpack("C*",$reddata);
        $green = unpack("C*",$greendata);
        $blue = unpack("C*",$bluedata);
        $size = count($red);
        $j=0;
        foreach(array_keys($red) as $i) {
            $buffer[$j++] = $red[$i];
            $buffer[$j++] = $green[$i];
            $buffer[$j++] = $blue[$i];
        }
    }
}
