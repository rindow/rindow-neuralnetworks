<?php
namespace RindowTest\NeuralNetworks\Dataset\Cifar10Test;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use SplFixedArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

class Cifar10Test extends TestCase
{
    protected $plot = false;
    protected $pickleFile;

    public function setUp() : void
    {
        $mo = new MatrixOperator();
        if(!$mo->isAdvanced()) {
            $this->markTestSkipped("The service is not Advanced.");
            return;
        }
        $this->plot = true;
        $this->pickleFile = $this->getDatasetDir().'/test_batch_labels.pkl';
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

    public function getPlotConfig()
    {
        return [
            'renderer.skipCleaning' => true,
            'renderer.skipRunViewer' => getenv('PLOT_RENDERER_SKIP') ? true : false,
        ];
    }

    public function testDownloadFiles()
    {
        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);

        $nn->datasets()->cifar10()->downloadFiles();
        $this->assertTrue(true);
    }

    public function testLoadDataFromFiles()
    {
        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $pickleFile = $this->pickleFile;
        if(file_exists($pickleFile)) {
            $nn->datasets()->cifar10()->cleanPickle();
            //unlink($pickleFile);
            sleep(1);
        }

        [$train,$test] =
            $nn->datasets()->cifar10()->loadData();

        sleep(1);
        $this->assertTrue(file_exists($pickleFile));

        if($this->plot) {
            foreach($train as [$img,$label]) {
                $this->assertEquals([10000,32, 32, 3],$img->shape());
                $this->assertEquals([10000],$label->shape());
                [$figure, $axes] = $plt->subplots(5,7);
                for($i=0;$i<count($axes);$i++) {
                    $axes[$i]->setAspect('equal');
                    $axes[$i]->setFrame(false);
                    $axes[$i]->imshow($img[$i],
                        null,null,null,$origin='upper');
                }
            }
            foreach($test as [$img,$label]) {
                $this->assertEquals([10000,32, 32, 3],$img->shape());
                $this->assertEquals([10000],$label->shape());
                [$figure, $axes] = $plt->subplots(5,7);
                for($i=0;$i<count($axes);$i++) {
                    $axes[$i]->setAspect('equal');
                    $axes[$i]->setFrame(false);
                    $axes[$i]->imshow($img[$i],
                        null,null,null,$origin='upper');
                }
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

        [$train,$test] =
            $nn->datasets()->cifar10()->loadData();

        sleep(1);
        $this->assertTrue(file_exists($pickleFile));

        if($this->plot) {
            foreach($train as [$img,$label]) {
                $this->assertEquals([10000,32, 32, 3],$img->shape());
                $this->assertEquals([10000],$label->shape());
                [$figure, $axes] = $plt->subplots(5,7);
                for($i=0;$i<count($axes);$i++) {
                    $axes[$i]->setAspect('equal');
                    $axes[$i]->setFrame(false);
                    $axes[$i]->imshow($img[$i],
                        null,null,null,$origin='upper');
                }
            }
            foreach($test as [$img,$label]) {
                $this->assertEquals([10000,32, 32, 3],$img->shape());
                $this->assertEquals([10000],$label->shape());
                [$figure, $axes] = $plt->subplots(5,7);
                for($i=0;$i<count($axes);$i++) {
                    $axes[$i]->setAspect('equal');
                    $axes[$i]->setFrame(false);
                    $axes[$i]->imshow($img[$i],
                        null,null,null,$origin='upper');
                }
            }
            $plt->show();
        }
    }

    public function testRewind()
    {
        $pickleFile = $this->pickleFile;
        $this->assertTrue(file_exists($pickleFile));

        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);

        [$train,$test] =
            $nn->datasets()->cifar10()->loadData();

        sleep(1);
        $this->assertTrue(file_exists($pickleFile));

        // first loop
        $i = 0;
        foreach($train as [$img,$label]) {
            $this->assertEquals([10000,32, 32, 3],$img->shape());
            $this->assertEquals([10000],$label->shape());
            $i++;
        }
        $this->assertEquals(5,$i);

        $i = 0;
        foreach($test as [$img,$label]) {
            $this->assertEquals([10000,32, 32, 3],$img->shape());
            $this->assertEquals([10000],$label->shape());
            $i++;
        }
        $this->assertEquals(1,$i);

        // rewind
        $i = 0;
        foreach($train as [$img,$label]) {
            $this->assertEquals([10000,32, 32, 3],$img->shape());
            $this->assertEquals([10000],$label->shape());
            $i++;
        }
        $this->assertEquals(5,$i);

        $i = 0;
        foreach($test as [$img,$label]) {
            $this->assertEquals([10000,32, 32, 3],$img->shape());
            $this->assertEquals([10000],$label->shape());
            $i++;
        }
        $this->assertEquals(1,$i);
    }

    public function testCleanPickle()
    {
        $pickleFile = $this->pickleFile;
        $this->assertTrue(file_exists($pickleFile));

        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);
        $nn->datasets()->cifar10()->cleanPickle();
        $this->assertTrue(true);
    }
}
