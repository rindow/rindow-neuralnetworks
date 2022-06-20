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
    public function newMatrixOperator()
    {
        return new MatrixOperator();
    }

    public function newNeuralNetworks($mo)
    {
        return new NeuralNetworks($mo);
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
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $this->assertInstanceof(
            'Rindow\NeuralNetworks\Loss\CategoricalCrossEntropy',
            $nn->losses()->CategoricalCrossEntropy());
    }

    public function testDefault()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $func = new CategoricalCrossEntropy($K);
/*
        $x = $K->array([
            [0.0, 0.0 , 6.0],
            [0.0, 0.0 , 6.0],
        ]);
        $t = $K->array([
            [0.0, 0.0 , 1.0],
            [0.0, 0.0 , 1.0],
        ]);
        $n = $x->shape()[0];

        $y = $backend->softmax($x);
        $loss = $func->forward($t,$y);
        $accuracy = $func->accuracy($t,$x);

        $this->assertLessThan(0.01,abs(0.0-$loss));

        $dx = $backend->dsoftmax($func->backward(1.0),$y);
        #$this->assertLessThan(0.0001, $K->scalar($K->asum($K->sub($K->sub($y,$dx),$t))));
*/
        #$x = $K->array([
        #    [0.0, 0.0 , 6.0],
        #    [0.0, 0.0 , 6.0],
        #]);
        #$t = $K->array([
        #    [0.0, 1.0 , 0.0],
        #    [0.0, 1.0 , 0.0],
        #]);
        $x = $K->array([
            [0.05, 0.95, 0], [0.1, 0.8, 0.1]
        ]);
        $t = $K->array([
            [0, 1, 0], [0, 0, 1]
        ]);

        $x = $K->softmax($x);
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($func,$t, $x) {
                $outputsVariable = $func->forward($t, $x);
                return $outputsVariable;
            }
        );
        $loss = $K->scalar($outputsVariable);
        $this->assertLessThan(0.01,abs(0.9868951-$loss));

        $dx = $outputsVariable->creator()->backward([$K->array(1.0)]);
        $dx = $K->dSoftmax($dx[1],$x);
        #$this->assertLessThan(0.001,$K->scalar($K->asum($K->sub($K->sub($x,$dx),$t))));
        $this->assertTrue($mo->la()->isclose(
            $mo->array([[ 0.11335728, -0.22118606,  0.10782879],
                        [ 0.12457169,  0.25085658, -0.3754283 ]]),
            $K->ndarray($dx)));
    }

    public function testFromLogits()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $func = new CategoricalCrossEntropy($K);
        $func->setFromLogits(true);
/*
        $x = $K->array([
            [0.0, 0.0 , 6.0],
            [0.0, 0.0 , 6.0],
        ]);
        $t = $K->array([
            [0.0, 0.0 , 1.0],
            [0.0, 0.0 , 1.0],
        ]);
        $y = $backend->softmax($x);

        $loss = $func->forward($t,$x);
        $this->assertLessThan(0.01,abs(0.0-$loss));

        $dx = $func->backward(1.0);
        $this->assertLessThan(0.0001,$K->scalar($K->asum($K->sub($K->sub($y,$dx),$t))));
*/
        #$x = $K->array([
        #    [0.0, 0.0 , 6.0],
        #    [0.0, 0.0 , 6.0],
        #]);
        #$t = $K->array([
        #    [0.0, 1.0 , 0.0],
        #    [0.0, 1.0 , 0.0],
        #]);
        $x = $K->array([
            [0.05, 0.95, 0], [0.1, 0.8, 0.1]
        ]);
        $t = $K->array([
            [0, 1, 0], [0, 0, 1]
        ]);
        //$y = $backend->softmax($x);

        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($func,$t, $x) {
                $outputsVariable = $func->forward($t, $x);
                return $outputsVariable;
            }
        );
        $loss = $K->scalar($outputsVariable);
        $this->assertLessThan(0.01,abs(0.9868951-$loss));

        $dx = $outputsVariable->creator()->backward([$K->array(1.0)]);
        $dx = $dx[1];
        #$this->assertLessThan(0.0001,$K->scalar($K->asum($K->sub($K->sub($y,$dx),$t))));
        $this->assertTrue($mo->la()->isclose(
            $mo->array([[ 0.11335728, -0.22118606,  0.10782879],
                        [ 0.12457169,  0.25085658, -0.3754283 ]]),
            $K->ndarray($dx)));
    }
}
