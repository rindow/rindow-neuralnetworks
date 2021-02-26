<?php
namespace RindowTest\NeuralNetworks\Dataset\FashionMnistTest;

if(!class_exists('RindowTest\NeuralNetworks\Dataset\MnistTest\Test')) {
    require_once __DIR__.'/MnistTest.php';
}

use RindowTest\NeuralNetworks\Dataset\MnistTest\Test as ORGTest;
use Interop\Polite\Math\Matrix\NDArray;
use SplFixedArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;


/**
 * @requires extension rindow_openblas
 */
class Test extends ORGTest
{
    protected $plot = false;
    protected $pickleFilename = '/rindow/nn/datasets/fashion-mnist/mnist.pkl';

    public function dataset($nn)
    {
        return $nn->datasets()->fashionMnist();
    }
}
