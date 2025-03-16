<?php
namespace RindowTest\NeuralNetworks\Gradient\Layer\LSTMTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

class LSTMTest extends TestCase
{
    public function newMatrixOperator()
    {
        return new MatrixOperator();
    }

    public function newNeuralNetworks($mo)
    {
        return new NeuralNetworks($mo);
    }

    public function newBackend($nn)
    {
        return $nn->backend();
    }

    public function testNormal()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x = $K->array([
            [0,1,2],
            [0,1,2],
        ],dtype:NDArray::int32);
        $x = $g->Variable($x);
        $embedding = $nn->layers->Embedding($inputDim=3, $outputDim=4, input_length:3);
        $layer = $nn->layers->LSTM($units=3);

        $outputs = $nn->with($tape=$g->GradientTape(),
            function() use ($embedding,$layer,$x) {
                $x1 = $embedding($x,true); // x1.shape=[2,3,4]
                $outputs = $layer($x1,true);
                return $outputs;
            }
        );
        $this->assertEquals([2,3],$outputs->value()->shape());
        $gradients = $tape->gradient($outputs, $layer->trainableVariables());

        $this->assertCount(3,$layer->trainableVariables());
        $this->assertCount(3,$gradients);
        $this->assertEquals([4,12],$gradients[0]->shape());
        $this->assertEquals([3,12],$gradients[1]->shape());
        $this->assertEquals([12],$gradients[2]->shape());
    }

    public function testWithInitialStates()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x = $K->array([
            [0,1,2],
            [0,1,2],
        ],dtype:NDArray::int32);
        $s = [
            $K->array([[0,1,2],[0,1,2],]),
            $K->array([[0,1,2],[0,1,2],]),
        ];
        $s = [
            $K->array([[0,1,2],[0,1,2],]),
            $K->array([[0,1,2],[0,1,2],]),
        ];
        $x = $g->Variable($x,name:'raw-x');
        $s = array_map(function($stat) use($g){return $g->Variable($stat,name:'raw-stat');},$s);
        $embed1 = $nn->layers->Embedding($inputDim=3, $outputDim=4, input_length:3);
        $flatten1 = $nn->layers->Flatten(input_shape:[3]);
        $flatten2 = $nn->layers->Flatten(input_shape:[3]);
        $layer = $nn->layers->LSTM($units=3);

        $outputs = $nn->with($tape=$g->GradientTape(),
            function() use ($embed1,$flatten1,$flatten2,$layer,$x,$s) {
                $x1 = $embed1($x,true); // x1.shape=[2,3,4]
                //$s1 = [$flatten1($s[0],true),$flatten2($s[1],true)];
                $s1 = [$flatten1($s[0],true),$flatten2($s[1],true)];
                $x1->setName('x');
                $s1[0]->setName('state1');
                $s1[1]->setName('state2');
                $outputs = $layer($x1,true,$s1);
                $outputs->setName('lstm_out');
                return $outputs;
            }
        );
        $this->assertEquals([2,3],$outputs->value()->shape());
        //$this->assertCount(2,$states);
        //$this->assertEquals([2,3],$states[0]->value()->shape());
        //$this->assertEquals([2,3],$states[1]->value()->shape());

        $gradients = $tape->gradient($outputs, array_merge($layer->weights(),$s));

        $this->assertCount(3,$layer->weights());
        $this->assertCount(5,$gradients);
        $this->assertEquals([4,12],$gradients[0]->shape());
        $this->assertEquals([3,12],$gradients[1]->shape());
        $this->assertEquals([12],$gradients[2]->shape());
        $this->assertEquals([2,3],$gradients[3]->shape());
        $this->assertEquals([2,3],$gradients[4]->shape());
    }

    public function testOutStates()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x = $K->array([
            [0,1,2],
            [0,1,2],
        ],dtype:NDArray::int32);
        $x = $g->Variable($x,name:'raw-x');
        $embed1 = $nn->layers->Embedding($inputDim=3, $outputDim=4, input_length:3);
        $layer = $nn->layers->LSTM($units=3,
            return_sequences:true,
            return_state:true,
        );

        [$outputs,$states] = $nn->with($tape=$g->GradientTape(),
            function() use ($embed1,$layer,$x) {
                $x1 = $embed1($x,true); // x1.shape=[2,3,4]
                $x1->setName('x');
                [$outputs,$states] = $layer($x1,true);
                $outputs->setName('lstm_out');
                return [$outputs,$states];
            }
        );
        $this->assertEquals([2,3,3],$outputs->value()->shape());
        $this->assertCount(2,$states);
        $this->assertEquals([2,3],$states[0]->value()->shape());
        $this->assertEquals([2,3],$states[1]->value()->shape());

        $gradients = $tape->gradient($outputs, $layer->weights());

        $this->assertEquals([4,12],$gradients[0]->shape());
        $this->assertEquals([3,12],$gradients[1]->shape());
        $this->assertEquals([12],$gradients[2]->shape());
    }

    public function testNoGradient()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x = $K->array([
            [0,1,2],
            [0,1,2],
        ],dtype:NDArray::int32);
        $x = $g->Variable($x,name:'raw-x');
        $embed1 = $nn->layers->Embedding($inputDim=3, $outputDim=4, input_length:3);
        $layer = $nn->layers->LSTM($units=3,
            return_sequences:true,
            return_state:true,
        );

        $x1 = $embed1($x,true); // x1.shape=[2,3,4]
        $x1->setName('x');
        [$outputs,$states] = $layer($x1,true);
        $outputs->setName('lstm_out');

        $this->assertEquals([2,3,3],$outputs->value()->shape());
        $this->assertCount(2,$states);
        $this->assertEquals([2,3],$states[0]->value()->shape());
        $this->assertEquals([2,3],$states[1]->value()->shape());
    }
}
