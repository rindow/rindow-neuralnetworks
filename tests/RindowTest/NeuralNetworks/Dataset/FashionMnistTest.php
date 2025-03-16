<?php
namespace RindowTest\NeuralNetworks\Dataset\FashionMnistTest;

if(!class_exists('RindowTest\NeuralNetworks\Dataset\MnistTest\MnistTest')) {
    require_once __DIR__.'/MnistTest.php';
}

use RindowTest\NeuralNetworks\Dataset\MnistTest\MnistTest as ORGTest;
use Interop\Polite\Math\Matrix\NDArray;
use SplFixedArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;


class FashionMnistTest extends ORGTest
{
    protected $plot = false;

    protected function getDatasetDir() : string
    {
        return $this->getRindowDatesetDir().'/fashion-mnist';
    }

    public function dataset($nn)
    {
        return $nn->datasets()->fashionMnist();
    }
}
