<?php
namespace RindowTest\NeuralNetworks\Activation\SoftmaxTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Activation\Softmax;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

class SoftmaxTest extends TestCase
{
    public function newMatrixOperator()
    {
        return new MatrixOperator();
    }

    public function newNeuralNetworks($mo)
    {
        return new NeuralNetworks($mo);
    }

    public function verifyGradient($mo, $K, $function, NDArray $x, ...$args)
    {
        $f = function($x) use ($K,$function,$args){
            $states = new \stdClass();
            $x = $K->array($x);
            $y = $function->forward($states,$x,...$args);
            return $K->ndarray($y);
        };
        $grads = $mo->la()->numericalGradient(1e-3,$f,$K->ndarray($x));
        $states = new \stdClass();
        $outputs = $function->forward($states,$x, ...$args);
        $ones = $K->ones($outputs->shape(),$outputs->dtype());
        $dInputs = $function->backward($states,$ones);
        return $mo->la()->isclose($grads[0],$K->ndarray($dInputs),null,1e-4);
    }

    public function testNormal()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $activation = new Softmax($K);

        $states = new \stdClass();
        $x = $mo->array([
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
        ]);
        $copyX = $mo->copy($x);
        $x = $K->array($x);
        $y = $activation->forward($states,$x);
        $y = $K->ndarray($y);
        $x = $K->ndarray($x);
        $this->assertEquals($copyX->toArray(),$x->toArray());
        $this->assertEquals($x->shape(),$y->shape());
        $this->assertTrue($mo->la()->isclose(
            $mo->ones([3]),
            $mo->sum($y,axis:1)
        ));

        $dout = $mo->array([
            [-0.5,-0.25,0.0,0.25,0.5],
            [-0.5,-0.25,0.0,0.25,0.5],
            [-0.5,-0.25,0.0,0.25,0.5],
        ]);
        $copydout = $mo->copy($dout);
        $dout = $K->array($dout);
        $dx = $activation->backward($states,$dout);
        $dx = $K->ndarray($dx);
        $dout = $K->ndarray($dout);
        $this->assertEquals($dout->shape(),$dx->shape());
        $this->assertEquals($copydout->toArray(),$dout->toArray());

        $inputs = $K->array([
            [-20.0,-15.0,0.0,5.0,10.0],
            [-10.0,-0.5,0.0,0.5,10.0],
            [-10.0,-0.5,0.0,0.5,10.0],
        ]);
        $this->assertTrue(
            $this->verifyGradient($mo,$K,$activation,$inputs));
    }

    public function testMask()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $activation = new Softmax($K);

        $states = new \stdClass();
        $x = $g->Variable([
            [[1,2,3,4],
             [1,2,3,4],
             [1,2,3,4]],
            [[1,2,3,4],
             [1,2,3,4],
             [1,2,3,4]],
        ]);

        $mask = $K->array([
            [1,0,0,0],
            [1,1,0,0],
            [1,1,1,1],
        ]);
        $copyX = $mo->copy($x);
        $x = $K->array($x);
        $y = $K->mul($x,$activation->forward($states,$x,mask:$mask));
        $y = $K->ndarray($y);
        $x = $K->ndarray($x);
        $this->assertEquals($copyX->toArray(),$x->toArray());
        $this->assertEquals($x->shape(),$y->shape());
        #echo $mo->toString($y,indent:true),"\n";
        $this->assertTrue($mo->la()->isclose(
            $y,
            $mo->array([
               [[1.,         0.,         0.,         0.,        ],
                [0.26894143, 1.4621172,  0.,         0.,        ],
                [0.0320586,  0.17428863, 0.7106484,  2.575657,  ]],
               [[1.,         0.,         0.,         0.,        ],
                [0.26894143, 1.4621172,  0.,         0.,        ],
                [0.0320586,  0.17428863, 0.7106484,  2.575657,  ]],
            ])
        ));

        #$dout = $mo->scale(0.5,$mo->ones([2,3,4]));
        $dout = $mo->copy($x);
        $copydout = $mo->copy($dout);
        $dout = $K->array($dout);
        $dx = $activation->backward($states,$dout);
        $dx = $K->ndarray($dx);
        $dout = $K->ndarray($dout);
        $this->assertEquals($dout->shape(),$dx->shape());
        $this->assertEquals($copydout->toArray(),$dout->toArray());
        #echo $mo->toString($dx,indent:true),"\n";
        $this->assertTrue($mo->la()->isclose(
            $dx,
            $mo->array([
                [[ 0.0,         0.0,         0.0,         0.0       ],
                 [-0.19661197,  0.19661197,  0.0,         0.0       ],
                 [-0.07991096, -0.13007617, -0.1167009,   0.32668802]],
                [[ 0.0,         0.0,         0.0,         0.0       ],
                 [-0.19661197,  0.19661197,  0.0,         0.0       ],
                 [-0.07991096, -0.13007617, -0.1167009,   0.32668802]],
            ])
        ));
 
        $inputs = $K->array([
            [-20.0,-15.0,0.0,5.0,10.0],
            [-10.0,-0.5,0.0,0.5,10.0],
            [-10.0,-0.5,0.0,0.5,10.0],
        ]);
        $this->assertTrue(
            $this->verifyGradient($mo,$K,$activation,$inputs));
    }
}
