<?php
namespace RindowTest\NeuralNetworks\Gradient\Func\SplitTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

class SplitTest extends TestCase
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
        $x = $g->Variable($x);


        //
        // dy[0]/dx = 1
        //
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x){
                $y = $g->split($x,[1,2]);
                return $y;
            }
        );

        $this->assertTrue($y[0]->value()->toArray()==[[1],[4]]);
        $this->assertTrue($y[1]->value()->toArray()==[[2,3],[5,6]]);

        $dx = $tape->gradient($y[0],$x);
        $this->assertEquals([
            [1,0,0],
            [1,0,0],
        ],$dx->toArray());


        //
        // dy[1]/dx = 1
        //
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x){
                $y = $g->split($x,[1,2]);
                return $y;
            }
        );

        $this->assertTrue($y[0]->value()->toArray()==[[1],[4]]);
        $this->assertTrue($y[1]->value()->toArray()==[[2,3],[5,6]]);

        $dx = $tape->gradient($y[1],$x);
        $this->assertEquals([
            [0,1,1],
            [0,1,1],
        ],$dx->toArray());
    }


    public function test3elementsNormal()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x = $K->array([
            [1,2,3,4,5,6],
            [7,8,9,10,11,12]
        ]);
        $x = $g->Variable($x);


        //
        // dy[0]/dx = 1
        //
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x){
                $y = $g->split($x,[2,2,2]);
                return $y;
            }
        );

        $this->assertTrue($y[0]->value()->toArray()==[[1,2],[7,8]]);
        $this->assertTrue($y[1]->value()->toArray()==[[3,4],[9,10]]);
        $this->assertTrue($y[2]->value()->toArray()==[[5,6],[11,12]]);

        $dx = $tape->gradient($y[0],$x);
        $this->assertEquals([
            [1,1,0,0,0,0],
            [1,1,0,0,0,0],
        ],$dx->toArray());


        //
        // dy[1]/dx = 1
        //
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x){
                $y = $g->split($x,[2,2,2]);
                return $y;
            }
        );

        $this->assertTrue($y[0]->value()->toArray()==[[1,2],[7,8]]);
        $this->assertTrue($y[1]->value()->toArray()==[[3,4],[9,10]]);
        $this->assertTrue($y[2]->value()->toArray()==[[5,6],[11,12]]);

        $dx = $tape->gradient($y[1],$x);
        $this->assertEquals([
            [0,0,1,1,0,0],
            [0,0,1,1,0,0],
        ],$dx->toArray());

        //
        // dy[2]/dx = 1
        //
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x){
                $y = $g->split($x,[2,2,2]);
                return $y;
            }
        );

        $this->assertTrue($y[0]->value()->toArray()==[[1,2],[7,8]]);
        $this->assertTrue($y[1]->value()->toArray()==[[3,4],[9,10]]);
        $this->assertTrue($y[2]->value()->toArray()==[[5,6],[11,12]]);

        $dx = $tape->gradient($y[2],$x);
        $this->assertEquals([
            [0,0,0,0,1,1],
            [0,0,0,0,1,1],
        ],$dx->toArray());
    }

    public function testWithAxisNormal()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x = $K->array([
            [1,2],
            [3,4],
            [5,6],
        ]);
        $x = $g->Variable($x);


        //
        // dy[0]/dx = 1
        //
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x){
                $y = $g->split($x,[1,2],axis:0);
                return $y;
            }
        );

        $this->assertTrue($y[0]->value()->toArray()==[[1,2]]);
        $this->assertTrue($y[1]->value()->toArray()==[[3,4],[5,6]]);

        $dx = $tape->gradient($y[0],$x);
        $this->assertEquals([
            [1,1],
            [0,0],
            [0,0],
        ],$dx->toArray());


        //
        // dy[1]/dx = 1
        //
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$x){
                $y = $g->split($x,[1,2],axis:0);
                return $y;
            }
        );

        $this->assertTrue($y[0]->value()->toArray()==[[1,2]]);
        $this->assertTrue($y[1]->value()->toArray()==[[3,4],[5,6]]);

        $dx = $tape->gradient($y[1],$x);
        $this->assertEquals([
            [0,0],
            [1,1],
            [1,1],
        ],$dx->toArray());
    }

}
