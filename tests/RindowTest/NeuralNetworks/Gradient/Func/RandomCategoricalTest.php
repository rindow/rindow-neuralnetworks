<?php
namespace RindowTest\NeuralNetworks\Gradient\Func\RandomCategoricalTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

class RandomCategoricalTest extends TestCase
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

    public function testSingleValue()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $logits = $g->Variable($K->log($K->array([1,2,3])));
        $action = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$logits){
                $action = $g->randomCategorical($logits);
                return $action;
            }
        );
        $this->assertEquals([],$action->shape());
        $this->assertEquals(NDArray::int32,$action->dtype());
        $this->assertTrue($logits->isbackpropagatable());
        $this->assertFalse($action->isbackpropagatable());
        try {
            $tape->gradient($action,$logits);
        } catch(\Throwable $e) {
            $error = $e->getMessage();
        }
        $this->assertStringStartsWith("No applicable gradient found for source",$error);
    }

    public function testMatrixValue()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $logits = $g->Variable($K->log($K->array([
            [3.0, 4.0],
            [1.0, 3.0],
            [1.0, 1.0],
        ])));
        $action = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$logits) {
                $action = $g->randomCategorical($logits);
                return $action;
            }
        );

        $this->assertEquals([3],$action->shape());
        $this->assertEquals(NDArray::int32,$action->dtype());
        $this->assertTrue($logits->isbackpropagatable());
        $this->assertFalse($action->isbackpropagatable());
        try {
            $tape->gradient($action,$logits);
        } catch(\Throwable $e) {
            $error = $e->getMessage();
        }
        $this->assertStringStartsWith("No applicable gradient found for source",$error);
    }

    public function testMuiltSamples()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $logits = $g->Variable($K->array([3.0, 4.0]));
        $action = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$logits) {
                $action = $g->randomCategorical($logits,batchShape:[3]);
                return $action;
            }
        );

        $this->assertEquals([3],$action->shape());
        $this->assertEquals(NDArray::int32,$action->dtype());
        $this->assertTrue($logits->isbackpropagatable());
        $this->assertFalse($action->isbackpropagatable());
        try {
            $tape->gradient($action,$logits);
        } catch(\Throwable $e) {
            $error = $e->getMessage();
        }
        $this->assertStringStartsWith("No applicable gradient found for source",$error);
    }

    public function testIlligalMuiltSamples()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('If batchShape is specified, the input must be one-dimensional.');
        
        $logits = $g->Variable($K->array([[3.0, 4.0]]));
        $action = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$logits) {
                $action = $g->randomCategorical($logits,batchShape:[3]);
                return $action;
            }
        );
    }

    public function testWithOptionValues()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $logits = $g->Variable($K->array([3.0, 4.0]));
        $action = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$logits) {
                $action = $g->randomCategorical(
                    $logits,
                    batchShape:[3],
                    softmax:false,
                    dtype:NDArray::int32,
                    seed:123,
                    name:'categorical',
                );
                return $action;
            }
        );

        $this->assertEquals([3],$action->shape());
        $this->assertEquals(NDArray::int32,$action->dtype());
        $this->assertTrue($logits->isbackpropagatable());
        $this->assertFalse($action->isbackpropagatable());
        try {
            $tape->gradient($action,$logits);
        } catch(\Throwable $e) {
            $error = $e->getMessage();
        }
        $this->assertStringStartsWith("No applicable gradient found for source",$error);
    }

}
