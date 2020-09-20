<?php
namespace RindowTest\NeuralNetworks\Layer\ActivationTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Layer\Activation;
use Rindow\NeuralNetworks\Activation\ReLU;
use Rindow\NeuralNetworks\Activation\Sigmoid;
use Rindow\NeuralNetworks\Activation\Softmax;
use Rindow\NeuralNetworks\Activation\Tanh;
use Rindow\NeuralNetworks\Loss\SparseCategoricalCrossEntropy;


class Test extends TestCase
{
    public function testResolveFunctions()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $activation = new Activation($backend,'tanh');
        $this->assertInstanceOf(Tanh::class,$activation->getActivation());
        $activation = new Activation($backend,'relu');
        $this->assertInstanceOf(ReLU::class,$activation->getActivation());
        $activation = new Activation($backend,'sigmoid');
        $this->assertInstanceOf(Sigmoid::class,$activation->getActivation());
        $activation = new Activation($backend,'softmax');
        $this->assertInstanceOf(Softmax::class,$activation->getActivation());
        
        $softmax = $activation->getActivation(); 
        $activation = new Activation($backend,$softmax);
        $this->assertInstanceOf(Softmax::class,$activation->getActivation());
    }

    public function testReplaceFunction()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $activation = new Activation($backend,'sigmoid');
        $this->assertInstanceOf(Sigmoid::class,$activation->getActivation());
        $loss = new SparseCategoricalCrossEntropy($backend);
        $activation->setActivation($loss);
        $this->assertInstanceOf(SparseCategoricalCrossEntropy::class,$activation->getActivation());
    }
    
    public function testNormalwithReLU()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $layer = new Activation($backend,'relu');
        $layer->build([5]);

        $inputs = $mo->array([
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
        ]);
        $copyInputs = $mo->copy($inputs);
        $outputs = $layer->forward($inputs, $training=true);
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
        $dInputs = $layer->backward($dOutputs);
        $this->assertEquals([4,5],$dInputs->shape());
        $this->assertEquals(
            [0.0,0.0,0.0,0.5,1.0],
            $dInputs[0]->toArray()
        );
        $this->assertEquals($copydOutputs->toArray(),$dOutputs->toArray());

    }
}
