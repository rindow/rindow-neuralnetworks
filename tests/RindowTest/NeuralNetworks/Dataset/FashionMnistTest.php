<?php
namespace RindowTest\NeuralNetworks\Dataset\FashionMnistTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use SplFixedArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

/**
 * @requires extension rindow_openblas
 */
class Test extends TestCase
{
    protected $plot = false;

    public function setUp() : void
    {
        $this->plot = true;
        $this->pickleFile = sys_get_temp_dir().'/rindow/nn/datasets/fashion-mnist/mnist.pkl';
    }

    public function testDownloadFiles()
    {
        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);

        $nn->datasets()->fashionMnist()->downloadFiles();
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
        $plot = new Plot(null,$mo);

        [[$train_img,$train_label],[$test_img,$test_label]] =
            $nn->datasets()->fashionMnist()->loadData();

        sleep(1);
        $this->assertTrue(file_exists($pickleFile));

        if($this->plot) {
            [$figure, $axes] = $plot->subplots(5,7);
            for($i=0;$i<count($axes);$i++) {
                $axes[$i]->setAspect('equal');
                $axes[$i]->setFrame(false);
                $axes[$i]->imshow($train_img[$i][0],null,null,null,$origin='upper');
            }
            $plot->show();
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
        $plot = new Plot(null,$mo);

        [[$train_img,$train_label],[$test_img,$test_label]] =
            $nn->datasets()->fashionMnist()->loadData();

        if($this->plot) {
            [$figure, $axes] = $plot->subplots(5,7);
            for($i=0;$i<count($axes);$i++) {
                $axes[$i]->setAspect('equal');
                $axes[$i]->setFrame(false);
                $axes[$i]->imshow($mo->op(255,'-',$train_img[$i][0]),'gray',null,null,$origin='upper');
            }
            $plot->show();
        }
    }

    public function testCleanPickle()
    {
        $pickleFile = $this->pickleFile;
        $this->assertTrue(file_exists($pickleFile));

        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);
        $nn->datasets()->fashionMnist()->cleanPickle();
        $this->assertTrue(true);
    }
}
