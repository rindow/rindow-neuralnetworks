<?php
namespace RindowTest\NeuralNetworks\Gradient\Func\TransposeTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

class TransposeTest extends TestCase
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

    public function testNormalWithoutPerm()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        /// reshape [] => [2,3,4,5]
        $x = $g->Variable($K->zeros([2,3,4,5]));
        $z = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x){
                $z = $g->transpose($x);
                return $z;
            }
        );
        $this->assertEquals([5,4,3,2],$z->shape());
        $dx = $tape->gradient($z,$x);
        $this->assertEquals([2,3,4,5],$dx->shape());
    }

    public function testNormalWithPerm()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        /// reshape [] => [2,3,4,5]
        $x = $g->Variable($K->zeros([2,3,4,5]));
        $z = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x){
                $z = $g->transpose($x,[0,1,3,2]);
                return $z;
            }
        );
        $this->assertEquals([2,3,5,4],$z->shape());
        $dx = $tape->gradient($z,$x);
        $this->assertEquals([2,3,4,5],$dx->shape());
    }

    public function testPermDuplicate()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $this->expectException(InvalidArgumentException::class);

        /// reshape [] => [2,3,4,5]
        $x = $g->Variable($K->zeros([2,3,4,5]));
        $z = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x){
                $z = $g->transpose($x,[0,1,1,2]);
                return $z;
            }
        );
    }
}
