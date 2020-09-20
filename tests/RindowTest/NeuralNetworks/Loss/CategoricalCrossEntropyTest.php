<?php
namespace RindowTest\NeuralNetworks\Loss\CategoricalCrossEntropyTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Loss\CategoricalCrossEntropy;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;

class Test extends TestCase
{
    public function getPlotConfig()
    {
        return [
            'renderer.skipCleaning' => true,
            'renderer.skipRunViewer' => getenv('TRAVIS_PHP_VERSION') ? true : false,
        ];
    }

    public function testBuilder()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $this->assertInstanceof(
            'Rindow\NeuralNetworks\Loss\CategoricalCrossEntropy',
            $nn->losses()->CategoricalCrossEntropy());
    }

    public function testDefault()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $func = new CategoricalCrossEntropy($backend);

        $x = $mo->array([
            [0.0, 0.0 , 6.0],
            [0.0, 0.0 , 6.0],
        ]);
        $t = $mo->array([
            [0.0, 0.0 , 1.0],
            [0.0, 0.0 , 1.0],
        ]);
        $y = $backend->softmax($x);
        $loss = $func->loss($t,$y);
        $accuracy = $func->accuracy($t,$x);

        $this->assertLessThan(0.01,abs(0.0-$loss));

        $dx = $backend->dsoftmax($func->differentiateLoss(),$y);
        $this->assertLessThan(0.0001,$mo->asum($mo->op($mo->op($y,'-',$dx),'-',$t)));


        $x = $mo->array([
            [0.0, 0.0 , 6.0],
            [0.0, 0.0 , 6.0],
        ]);
        $t = $mo->array([
            [0.0, 1.0 , 0.0],
            [0.0, 1.0 , 0.0],
        ]);
        $y = $backend->softmax($x);
        $loss = $func->loss($t,$y);
        $this->assertLessThan(0.01,abs(6.0-$loss));

        $dx = $backend->dsoftmax($func->differentiateLoss(),$y);
        $this->assertLessThan(0.001,$mo->asum($mo->op($mo->op($y,'-',$dx),'-',$t)));
    }

    public function testFromLogits()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $func = new CategoricalCrossEntropy($backend);
        $func->setFromLogits(true);

        $x = $mo->array([
            [0.0, 0.0 , 6.0],
            [0.0, 0.0 , 6.0],
        ]);
        $t = $mo->array([
            [0.0, 0.0 , 1.0],
            [0.0, 0.0 , 1.0],
        ]);
        $y = $func->forward($x,true);
        $loss = $func->loss($t,$y);
        $this->assertLessThan(0.01,abs(0.0-$loss));

        $dx = $func->differentiateLoss();
        $this->assertLessThan(0.0001,$mo->asum($mo->op($mo->op($y,'-',$dx),'-',$t)));

        $x = $mo->array([
            [0.0, 0.0 , 6.0],
            [0.0, 0.0 , 6.0],
        ]);
        $t = $mo->array([
            [0.0, 1.0 , 0.0],
            [0.0, 1.0 , 0.0],
        ]);
        $y = $func->forward($x,true);
        $loss = $func->loss($t,$y);
        $this->assertLessThan(0.01,abs(6.0-$loss));

        $dx = $func->differentiateLoss();
        $this->assertLessThan(0.0001,$mo->asum($mo->op($mo->op($y,'-',$dx),'-',$t)));
    }
}
