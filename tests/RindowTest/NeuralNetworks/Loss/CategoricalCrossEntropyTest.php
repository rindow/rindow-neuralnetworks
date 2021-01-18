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
    public function newBackend($mo)
    {
        $builder = new NeuralNetworks($mo);
        return $builder->backend();
    }

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
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $this->assertInstanceof(
            'Rindow\NeuralNetworks\Loss\CategoricalCrossEntropy',
            $nn->losses()->CategoricalCrossEntropy());
    }

    public function testDefault()
    {
        $mo = new MatrixOperator();
        $K = $backend = $this->newBackend($mo);
        $func = new CategoricalCrossEntropy($backend);

        $x = $K->array([
            [0.0, 0.0 , 6.0],
            [0.0, 0.0 , 6.0],
        ]);
        $t = $K->array([
            [0.0, 0.0 , 1.0],
            [0.0, 0.0 , 1.0],
        ]);
        $y = $backend->softmax($x);
        $loss = $func->loss($t,$y);
        $accuracy = $func->accuracy($t,$x);

        $this->assertLessThan(0.01,abs(0.0-$loss));

        $dx = $backend->dsoftmax($func->differentiateLoss(),$y);
        $this->assertLessThan(0.0001, $K->scalar($K->asum($K->sub($K->sub($y,$dx),$t))));


        $x = $K->array([
            [0.0, 0.0 , 6.0],
            [0.0, 0.0 , 6.0],
        ]);
        $t = $K->array([
            [0.0, 1.0 , 0.0],
            [0.0, 1.0 , 0.0],
        ]);
        $y = $backend->softmax($x);
        $loss = $func->loss($t,$y);
        $this->assertLessThan(0.01,abs(6.0-$loss));

        $dx = $backend->dsoftmax($func->differentiateLoss(),$y);
        $this->assertLessThan(0.001,$K->scalar($K->asum($K->sub($K->sub($y,$dx),$t))));
    }

    public function testFromLogits()
    {
        $mo = new MatrixOperator();
        $K = $backend = $this->newBackend($mo);
        $func = new CategoricalCrossEntropy($backend);
        $func->setFromLogits(true);

        $x = $K->array([
            [0.0, 0.0 , 6.0],
            [0.0, 0.0 , 6.0],
        ]);
        $t = $K->array([
            [0.0, 0.0 , 1.0],
            [0.0, 0.0 , 1.0],
        ]);
        $y = $func->forward($x,true);
        $loss = $func->loss($t,$y);
        $this->assertLessThan(0.01,abs(0.0-$loss));

        $dx = $func->differentiateLoss();
        $this->assertLessThan(0.0001,$K->scalar($K->asum($K->sub($K->sub($y,$dx),$t))));

        $x = $K->array([
            [0.0, 0.0 , 6.0],
            [0.0, 0.0 , 6.0],
        ]);
        $t = $K->array([
            [0.0, 1.0 , 0.0],
            [0.0, 1.0 , 0.0],
        ]);
        $y = $func->forward($x,true);
        $loss = $func->loss($t,$y);
        $this->assertLessThan(0.01,abs(6.0-$loss));

        $dx = $func->differentiateLoss();
        $this->assertLessThan(0.0001,$K->scalar($K->asum($K->sub($K->sub($y,$dx),$t))));
    }
}
