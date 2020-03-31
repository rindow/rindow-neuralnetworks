<?php
namespace RindowTest\NeuralNetworks\Dataset\MnistTest;

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
    }

    public function testDownloadFiles()
    {
        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);

        $nn->datasets()->mnist()->downloadFiles();
        $this->assertTrue(true);
    }

    public function testLoadDataFromFiles()
    {
        $pickleFile = sys_get_temp_dir().'/rindow/nn/datasets/mnist.pkl';
        if(file_exists($pickleFile)) {
            unlink($pickleFile);
            sleep(1);
        }

        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);
        $plot = new Plot(null,$mo);

        [[$train_img,$train_label],[$test_img,$test_label]] =
            $nn->datasets()->mnist()->loadData();

        sleep(1);
        $this->assertTrue(file_exists($pickleFile));

        if($this->plot) {
            [$figure, $axes] = $plot->subplots(5,7);
            for($i=0;$i<count($axes);$i++) {
                $axes[$i]->setAspect('equal');
                $axes[$i]->setFrame(false);
                $axes[$i]->imshow($train_img[$i][0]);
            }
            $plot->show();
        }
    }

    public function testLoadDataFromPickle()
    {
        $pickleFile = sys_get_temp_dir().'/rindow/nn/datasets/mnist.pkl';
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
            $nn->datasets()->mnist()->loadData();

        if($this->plot) {
            [$figure, $axes] = $plot->subplots(5,7);
            for($i=0;$i<count($axes);$i++) {
                $axes[$i]->setAspect('equal');
                $axes[$i]->setFrame(false);
                $axes[$i]->imshow($mo->op(255,'-',$train_img[$i][0]),'gray');
            }
            $plot->show();
        }
    }

    public function testCleanPickle()
    {
        $pickleFile = sys_get_temp_dir().'/rindow/nn/datasets/mnist.pkl';
        $this->assertTrue(file_exists($pickleFile));

        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);
        $nn->datasets()->mnist()->cleanPickle();
        $this->assertTrue(true);
    }
}
