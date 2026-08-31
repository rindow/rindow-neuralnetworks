<?php
namespace RindowTest\NeuralNetworks\Gradient\Func\ConcatTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

class ConcatTest extends TestCase
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
            [1,2,3],
            [4,5,6],
        ]);
        $y = $K->array([
            [7,8,9],
            [10,11,12],
        ]);
        $salt = $K->increment($K->concat([$x,$y]),100);

        $x = $g->Variable($x);
        $y = $g->Variable($y);
        $salt = $g->Variable($salt);

        [$z,$output] = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x,$y,$salt){
                $z = $g->concat([$x,$y]);
                $output = $g->mul($z,$salt);
                return [$z,$output];
            }
        );

        $this->assertTrue($z->shape()==[2,6]);
        
        $this->assertEquals([
            [1,2,3,7,8,9],
            [4,5,6,10,11,12],
        ],$z->toArray());

        [$dx,$dy] = $tape->gradient($output,[$x,$y]);
        $this->assertEquals([
            [101,102,103],
            [104,105,106],
        ],$dx->toArray());
        $this->assertEquals([
            [107,108,109],
            [110,111,112],
        ],$dy->toArray());

    }


    public function test3elementsNormal()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x = $K->array([
            [1,2,3],
            [4,5,6],
        ]);
        $y = $K->array([
            [7,8,9],
            [10,11,12],
        ]);
        $z = $K->array([
            [13,14,15],
            [16,17,18],
        ]);
        $salt = $K->increment($K->concat([$x,$y,$z]),100);

        $x = $g->Variable($x);
        $y = $g->Variable($y);
        $z = $g->Variable($z);
        $salt = $g->Variable($salt);

        [$zz,$output] = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x,$y,$z,$salt){
                $zz = $g->concat([$x,$y,$z]);
                $output = $g->mul($zz,$salt);
                return [$zz,$output];
            }
        );

        $this->assertTrue($zz->shape()==[2,9]);
        
        $this->assertEquals([
            [1,2,3,7,8,9,13,14,15],
            [4,5,6,10,11,12,16,17,18],
        ],$zz->toArray());

        [$dx,$dy,$dz] = $tape->gradient($output,[$x,$y,$z]);
        $this->assertEquals([
            [101,102,103],
            [104,105,106],
        ],$dx->toArray());
        $this->assertEquals([
            [107,108,109],
            [110,111,112],
        ],$dy->toArray());
        $this->assertEquals([
            [113,114,115],
            [116,117,118],
        ],$dz->toArray());

    }

    public function testWithAxisNormal()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x = $K->array([
            [1,2,3],
            [4,5,6],
        ]);
        $y = $K->array([
            [7,8,9],
            [10,11,12],
        ]);
        $salt = $K->increment($K->concat([$x,$y],axis:0),100);

        $x = $g->Variable($x);
        $y = $g->Variable($y);
        $salt = $g->Variable($salt);

        [$z,$output] = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x,$y,$salt){
                $z = $g->concat([$x,$y],axis:0);
                $output = $g->mul($z,$salt);
                return [$z,$output];
            }
        );

        $this->assertTrue($z->shape()==[4,3]);
        
        $this->assertEquals([
            [1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,12],
        ],$z->toArray());

        [$dx,$dy] = $tape->gradient($output,[$x,$y]);
        $this->assertEquals([
            [101,102,103],
            [104,105,106],
        ],$dx->toArray());
        $this->assertEquals([
            [107,108,109],
            [110,111,112],
        ],$dy->toArray());

    }

}
