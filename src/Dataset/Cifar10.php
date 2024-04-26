<?php
namespace Rindow\NeuralNetworks\Dataset;

use LogicException;
use RuntimeException;
use Rindow\Math\Matrix\MatrixOperator;
use Interop\Polite\Math\Matrix\NDArray;
use PharData;
use function Rindow\Math\Matrix\R;

class Cifar10
{
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
    protected array $imageShape = [32, 32, 3]; // = 784
    protected string $datasetDir;
    protected string $saveFile;

    public function __construct(object $mo)
    {
        $this->matrixOperator = $mo;
        $this->datasetDir = $this->getDatasetDir();
        if(!file_exists($this->datasetDir)) {
            @mkdir($this->datasetDir,0777,true);
        }
        $this->saveFile = $this->datasetDir . "/cifar10.pkl";
    }

    protected function getDatasetDir() : string
    {
        return sys_get_temp_dir().'/rindow/nn/datasets/cifar-10-batches-bin';
    }

    protected function console(string $message) : void
    {
        fwrite(STDERR,$message);
    }

    /**
     * @return array{array{NDArray,NDArray},array{NDArray,NDArray}}
     */
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

    public function cleanPickle(string $filePath=null) : void
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

        if(file_exists($this->datasetDir.'/data_batch_1.bin')){
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

    /**
     * @return array<string,NDArray>
     */
    protected function convertNDArray() : array
    {
        $mo = $this->matrixOperator;
        $filenames = $this->keyFiles;
        $testFiles = [array_pop($filenames)];
        $dataset['train_images'] = $mo->zeros([50000,32,32,3],NDArray::uint8);
        $dataset['train_labels'] = $mo->zeros([50000],NDArray::uint8);
        $this->convertDataset(
            $filenames,
            $dataset['train_images'],
            $dataset['train_labels']);
        $dataset['test_images'] = $mo->zeros([10000,32,32,3],NDArray::uint8);
        $dataset['test_labels'] = $mo->zeros([10000],NDArray::uint8);
        $this->convertDataset(
            $testFiles,
            $dataset['test_images'],
            $dataset['test_labels']);
        return $dataset;
    }

    /**
     * @param array<string> $filenames
     */
    protected function convertDataset(
        array $filenames,
        NDArray $image_dataset,
        NDArray $labels_dataset
        ) : void
    {
        $shape = $image_dataset->shape();
        $rows = array_shift($shape);
        $size = array_product($shape);
        $image_dataset = $image_dataset->reshape([$rows,$size]);
        $labels_dataset = $labels_dataset->reshape([$labels_dataset->size(),1]);
        $offset = 0;
        foreach($filenames as $filename) {
            $images = $image_dataset[R($offset,$offset+10000)];
            $labels = $labels_dataset[R($offset,$offset+10000)];
            $this->convertImage(
                $filename, $images, $labels
            );
            $offset += 10000;
        }
    }

    protected function convertImage(
        string $filename,
        NDArray $images,
        NDArray $labels
        ) : void
    {
        $mo = $this->matrixOperator;
        $filePath = $this->datasetDir."/".$filename;
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
            if($label===false||$label=='')
                break;
            if($labels->size()<=$p){
                throw new RuntimeException('label buffer overflow');
            }
            $this->unpackLabel(
                $label,
                $labels[$p]);
            $red = fread($f,1024);
            $green = fread($f,1024);
            $blue = fread($f,1024);
            if($red===false||
                $green===false||
                $blue===false)
                break;
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
