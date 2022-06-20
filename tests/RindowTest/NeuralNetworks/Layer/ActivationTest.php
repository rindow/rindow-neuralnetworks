<?php
namespace RindowTest\NeuralNetworks\Layer\ActivationTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Layer\Activation;
use Rindow\NeuralNetworks\Activation\ReLU;
use Rindow\NeuralNetworks\Activation\Sigmoid;
use Rindow\NeuralNetworks\Activation\Softmax;
use Rindow\NeuralNetworks\Activation\Tanh;
use Rindow\NeuralNetworks\Loss\SparseCategoricalCrossEntropy;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

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

    public function testResolveFunctions()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $activation = new Activation($K,'tanh');
        $this->assertInstanceOf(Tanh::class,$activation->getActivation());
        $activation = new Activation($K,'relu');
        $this->assertInstanceOf(ReLU::class,$activation->getActivation());
        $activation = new Activation($K,'sigmoid');
        $this->assertInstanceOf(Sigmoid::class,$activation->getActivation());
        $activation = new Activation($K,'softmax');
        $this->assertInstanceOf(Softmax::class,$activation->getActivation());

        $softmax = $activation->getActivation();
        $activation = new Activation($K,$softmax);
        $this->assertInstanceOf(Softmax::class,$activation->getActivation());
    }

    /*
        Loss Function has changed the function of Activation.

    public function testReplaceFunction()
    {
        $mo = $this->newMatrixOperator();
        $K = $this->newBackend($mo);
        $activation = new Activation($K,'sigmoid');
        $this->assertInstanceOf(Sigmoid::class,$activation->getActivation());
        $loss = new SparseCategoricalCrossEntropy($K);
        $activation->setActivation($loss);
        $this->assertInstanceOf(SparseCategoricalCrossEntropy::class,$activation->getActivation());
    }
    */

    public function testNormalwithReLU()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $layer = new Activation($K,'relu');
        $inputs = $g->Variable($K->zeros([1,5]));
        $layer->build($inputs);

        $inputs = $mo->array([
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
        ]);
        $copyInputs = $mo->copy($inputs);
        $inputs = $K->array($inputs);
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($layer,$inputs) {
                $outputsVariable = $layer->forward($inputs, $training=true);
                return $outputsVariable;
            }
        );
        $outputs = $K->ndarray($outputsVariable);
        $inputs = $K->ndarray($inputs);
        $this->assertEquals([4,5],$outputs->shape());
        $this->assertEquals(
            [0.0,0.0,0.0,0.5,1.0],
            $outputs[0]->toArray()
        );
        $this->assertEquals($copyInputs->toArray(),$inputs->toArray());


        $dOutputs = $mo->array([
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
        ]);
        $copydOutputs = $mo->copy($dOutputs);
        $dOutputs = $K->array($dOutputs);
        [$dInputs] = $outputsVariable->creator()->backward([$dOutputs]);
        $dInputs = $K->ndarray($dInputs);
        $dOutputs = $K->ndarray($dOutputs);
        $this->assertEquals([4,5],$dInputs->shape());
        $this->assertEquals(
            [0.0,0.0,0.0,0.5,1.0],
            $dInputs[0]->toArray()
        );
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());

    }
}
