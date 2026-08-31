<?php
namespace RindowTest\NeuralNetworks\Dataset\MnistTest;

if(class_exists('RindowTest\NeuralNetworks\Dataset\MnistTest\MnistTest')) {
    return;
}

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use SplFixedArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

class MnistTest extends TestCase
{
    protected $plot = false;
    protected $pickleFilename = '/mnist.pkl';
    protected $pickleFile;

    public function setUp() : void
    {
        parent::setUp();
        if(getenv('BLOBDATA_TEST_SKIP')) {
            $this->markTestSkipped("The service is not Advanced.");
            return;
        }
        $this->plot = true;
        $this->pickleFile = $this->getDatasetDir().$this->pickleFilename;
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

    public function getPlotConfig()
    {
        return [
            'renderer.skipCleaning' => true,
            'renderer.skipRunViewer' => getenv('PLOT_RENDERER_SKIP') ? true : false,
        ];
    }

    public function dataset($nn)
    {
        return $nn->datasets()->mnist();
    }

    public function testDownloadFiles()
    {
        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);

        $this->dataset($nn)->downloadFiles();
        $this->assertTrue(true);
    }

    public function testLoadDataFromFiles()
    {
        $pickleFile = $this->pickleFile;
        if(file_exists($pickleFile)) {
            unlink($pickleFile);
            sleep(1);
        }

        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);
        $plt = new Plot($this->getPlotConfig(),$mo);

        [[$train_img,$train_label],[$test_img,$test_label]] =
            $this->dataset($nn)->loadData();

        sleep(1);
        $this->assertTrue(file_exists($pickleFile));

        if($this->plot) {
            [$figure, $axes] = $plt->subplots(5,7);
            for($i=0;$i<count($axes);$i++) {
                $axes[$i]->setAspect('equal');
                $axes[$i]->setFrame(false);
                $axes[$i]->imshow($train_img[$i]->reshape([28,28]),null,null,null,$origin='upper');
            }
            $plt->show();
        }
    }

    public function testLoadDataFromPickle()
    {
        $pickleFile = $this->pickleFile;
        $this->assertTrue(file_exists($pickleFile));

        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);
        //$config = [
        //    'figure.bgColor' => 'white',
        //    'figure.figsize' => [500,500],
        //    'figure.leftMargin' => 0,
        //    'figure.bottomMargin' => 0,
        //    'figure.rightMargin' => 0,
        //    'figure.topMargin' => 0,
        //];
        $plt = new Plot($this->getPlotConfig(),$mo);

        [[$train_img,$train_label],[$test_img,$test_label]] =
            $this->dataset($nn)->loadData();

        if($this->plot) {
            [$figure, $axes] = $plt->subplots(5,7);
            for($i=0;$i<count($axes);$i++) {
                $axes[$i]->setAspect('equal');
                $axes[$i]->setFrame(false);
                $axes[$i]->imshow($mo->op(255,'-',$train_img[$i]->reshape([28,28])),'gray',null,null,$origin='upper');
            }
            $plt->show();
        }
    }

    public function testCleanPickle()
    {
        $pickleFile = $this->pickleFile;
        $this->assertTrue(file_exists($pickleFile));

        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);
        $this->dataset($nn)->cleanPickle();
        $this->assertTrue(true);
    }
}
