<?php
namespace RindowTest\NeuralNetworks\Gradient\Core\GraphFunctionTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Gradient\Core\GraphFunction;
use Rindow\NeuralNetworks\Gradient\Core\Variable;

class TestClass
{
    protected $nn;
    protected $K;
    protected $g;
    protected $sessionMode;

    public function __construct($nn)
    {
        $this->nn = $nn;
        $this->K = $nn->backend();
        $this->g = $nn->gradient();
    }

    public function setCheckSessionMode($mode)
    {
        $this->sessionMode = $mode;
    }

    public function testfunc($a, $x)
    {
        $nn = $this->nn;
        $K = $this->K;
        $g = $this->g;

        $b = $nn->gradient->Variable($K->array(2)); // constant

        assert($this->sessionMode==GraphFunction::$mode,'session mode check');
        if(GraphFunction::$mode==GraphFunction::EXECUTING) {
            assert($a instanceof NDArray,'value class:'.get_class($a));
            assert($x instanceof NDArray,'value class:'.get_class($x));
            assert($b instanceof NDArray,'value class:'.get_class($b));
        } else {
            assert(Variable::class==get_class($a),'value class:'.get_class($a));
            assert(Variable::class==get_class($x),'value class:'.get_class($x));
            assert(Variable::class==get_class($b),'value class:'.get_class($b));
        }

        $c = $g->add($g->mul($a,$x),$b);

        if(GraphFunction::$mode==GraphFunction::EXECUTING) {
            assert($c instanceof NDArray,'value class:'.get_class($c));
        } else {
            assert(Variable::class==get_class($c),'value class:'.get_class($c));
        }
        return $c;
    }
}

class GraphFunctionTest extends TestCase
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

    public function testFunctionForwardNormal()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $a = $nn->gradient->Variable($K->array(3));
        $x = $nn->gradient->Variable($K->array(4));
        $testclass = new TestClass($nn);
        $func = $g->Function([$testclass,'testfunc']);

        $testclass->setCheckSessionMode(GraphFunction::UNDER_CONSTRUCTION);
        $y = $func($a,$x);
        $this->assertEquals("14",$mo->toString($y->value()));

        $a = $nn->gradient->Variable($K->array(5));
        $x = $nn->gradient->Variable($K->array(6));
        $testclass->setCheckSessionMode(GraphFunction::EXECUTING);
        $y = $func($a,$x);
        $this->assertEquals("32",$mo->toString($y->value()));
    }

    public function testFunctionBackwardNormal()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $a = $nn->gradient->Variable($K->array(3));
        $x = $nn->gradient->Variable($K->array(4));
        $testclass = new TestClass($nn);
        $func = $g->Function([$testclass,'testfunc']);

        // Not yet built function
        $testclass->setCheckSessionMode(GraphFunction::UNDER_CONSTRUCTION);
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($func,$a,$x) {
                $y = $func($a,$x);
                return $y;
            }
        );
        $this->assertEquals("14",$mo->toString($y->value()));
        $grads = $tape->gradient($y,[$a,$x]);
        $this->assertCount(2,$grads);
        $this->assertEquals("4",$mo->toString($grads[0]));
        $this->assertEquals("3",$mo->toString($grads[1]));

        // built function
        $a = $nn->gradient->Variable($K->array(5));
        $x = $nn->gradient->Variable($K->array(6));
        $testclass->setCheckSessionMode(GraphFunction::EXECUTING);
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($func,$a,$x) {
                $y = $func($a,$x);
                return $y;
            }
        );
        $this->assertEquals("32",$mo->toString($y->value()));
        $grads = $tape->gradient($y,[$a,$x]);
        $this->assertCount(2,$grads);
        $this->assertEquals("6",$mo->toString($grads[0]));
        $this->assertEquals("5",$mo->toString($grads[1]));
    }

    public function testFunctionBackwardFirstNoGrad()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $testclass = new TestClass($nn);
        $func = $g->Function([$testclass,'testfunc']);

        // Not yet built function
        $a = $nn->gradient->Variable($K->array(3));
        $x = $nn->gradient->Variable($K->array(4));
        $testclass->setCheckSessionMode(GraphFunction::UNDER_CONSTRUCTION);
        $y = $func($a,$x);
        $this->assertEquals("14",$mo->toString($y->value()));

        // built function
        $a = $nn->gradient->Variable($K->array(5));
        $x = $nn->gradient->Variable($K->array(6));
        $testclass->setCheckSessionMode(GraphFunction::EXECUTING);
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($func,$a,$x) {
                $y = $func($a,$x);
                return $y;
            }
        );
        $this->assertEquals("32",$mo->toString($y->value()));
        $grads = $tape->gradient($y,[$a,$x]);
        $this->assertCount(2,$grads);
        $this->assertEquals("6",$mo->toString($grads[0]));
        $this->assertEquals("5",$mo->toString($grads[1]));
    }

    public function testSortManyOutputFunc()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $a = $g->Variable($K->array(3));
        $b = $g->Variable($K->array(4));
        $func = $g->Function(function($a,$b) use ($g) {
            $x = $g->mul($a,$b);  // 3*4 = 12
            $y = $g->add($x,$a);  // 12+3 = 15
            $c = $g->sub($x,$y);  // 12-15 = -3
            return [$c,$y];
        });
        [$c,$y] = $func($a,$b);
        $this->assertEquals(-3,$c->value()->toArray());
        $this->assertEquals(15,$y->value()->toArray());
        [$c,$y] = $func($a,$b);
        $this->assertEquals(-3,$c->value()->toArray());
        $this->assertEquals(15,$y->value()->toArray());
    }

    public function testNestGraphForwardNormal()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $func0 = $g->Function(function($a,$b) use ($K,$g) {
            $cc = $g->Variable($K->array(4));
            $x = $g->mul($a,$b);   // 6*2 = 12
            $y = $g->add($x,$cc);  // 12+4 = 16
            return $y;
        });
        $func1 = $g->Function(function($b,$c) use ($K,$g,$func0) {
            $c0 = $g->add($c,$c);  // 3+3 = 6
            $x = $func0($c0,$b);   // fn(6,2) = 16
            $y = $g->mul($x,$c);   // 16*3 = 48
            return $y;
        });

        $b = $g->Variable($K->array(2));
        $c = $g->Variable($K->array(3));

        $y = $func1($b,$c);
        $this->assertEquals(48,$y->value()->toArray());
        $y = $func1($b,$c);
        $this->assertEquals(48,$y->value()->toArray());
    }

    public function testNestGraphForwardCallBuiltGraph()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $func0 = $g->Function(function($a,$b) use ($K,$g) {
            $cc = $g->Variable($K->array(4));
            $x = $g->mul($a,$b);   // 6*2 = 12
            $y = $g->add($x,$cc);  // 12+4 = 16
            return $y;
        });
        $func1 = $g->Function(function($b,$c) use ($K,$g,$func0) {
            $c0 = $g->add($c,$c);  // 3+3 = 6
            $x = $func0($c0,$b);   // fn(6,2) = 16
            $y = $g->mul($x,$c);   // 16*3 = 48
            return $y;
        });

        // compile func0 only
        $a = $g->Variable($K->array(2));
        $b = $g->Variable($K->array(3));
        $y = $func0($a,$b);

        $b = $g->Variable($K->array(2));
        $c = $g->Variable($K->array(3));

        // compile func1 with compiled func0
        $y = $func1($b,$c);
        $this->assertEquals(48,$y->value()->toArray());
        $y = $func1($b,$c);
        $this->assertEquals(48,$y->value()->toArray());
    }

    public function testNestGraphBackwardNormal()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $func0 = $g->Function(function($a,$b) use ($K,$g) {
                                    // a=6 c=2         : da=6 db=18
            $cc = $g->Variable($K->array(4));
            $x = $g->mul($a,$b);    // 6*2 = 12        : da=2*3=6  db=6*3=18
            $y = $g->add($x,$cc);   // 12+4 = 16       : dx=3      dcc=3
            return $y;              //                 : dy = 3 
        });
        $func1 = $g->Function(function($b,$c) use ($K,$g,$func0) {
                                    // b=2 c=3         : db=18 dc=28
            $c0 = $g->add($c,$c);   // 3+3 = 6         : dc(16)+=6+6=28
            $x = $func0($c0,$b);    // fn(6,2) = 16    : dc0=6     db=18
            $y = $g->mul($x,$c);    // 16*3 = 48       : dx=1*3=3  dc=1*16=16
            return $y;              //                 : dy=1
        });

        $b = $g->Variable($K->array(2));
        $c = $g->Variable($K->array(3));

        // building
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($func1,$b,$c) {
                $y = $func1($b,$c);
                return $y;
            }
        );
        $this->assertEquals(48,$y->value()->toArray());
        $grads = $tape->gradient($y,[$b,$c]);
        $this->assertCount(2,$grads);
        $this->assertEquals(18,$grads[0]->toArray()); // db
        $this->assertEquals(28,$grads[1]->toArray()); // dc
 
        // built graph
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($func1,$b,$c) {
                $y = $func1($b,$c);
                return $y;
            }
        );
        $this->assertEquals(48,$y->value()->toArray());
        $grads = $tape->gradient($y,[$b,$c]);
        $this->assertCount(2,$grads);
        $this->assertEquals(18,$grads[0]->toArray());
        $this->assertEquals(28,$grads[1]->toArray());
    }

    public function testLayerInGraphNormal()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $layer0 = $nn->layers->Dense(3, kernel_initializer:'ones');
        $func0 = $g->Function(function($a,$b) use ($K,$g,$layer0) {
            $x = $g->add($a,$b);
            $y = $layer0($x,true);
            return $y;
        });

        $a = $g->Variable($K->array([[2],[3]]));
        $b = $g->Variable($K->array([[2],[4]]));

        // building
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($func0,$a,$b) {
                $y = $func0($a,$b);
                return $y;
            }
        );
        $this->assertEquals([2,3],$y->value()->shape());
        $grads = $tape->gradient($y,array_merge([$a,$b],$layer0->weights()));
        $this->assertCount(4,$grads);
        $this->assertEquals([2,1],$grads[0]->shape());
        $this->assertEquals([2,1],$grads[1]->shape());
        $this->assertEquals([1,3],$grads[2]->shape());
        $this->assertEquals([3],$grads[3]->shape());

        $this->assertEquals('[[4,4,4],[7,7,7]]',$K->toString($y->value()));
        $this->assertEquals('[[3],[3]]',$K->toString($grads[0]));
        $this->assertEquals('[[3],[3]]',$K->toString($grads[1]));
        $this->assertEquals('[[11,11,11]]',$K->toString($grads[2]));
        $this->assertEquals('[2,2,2]',$K->toString($grads[3]));

        $a = $g->Variable($K->array([[2],[3],[3]]));
        $b = $g->Variable($K->array([[2],[4],[3]]));
        // built graph
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($func0,$a,$b) {
                $y = $func0($a,$b);
                return $y;
            }
        );
        $this->assertEquals([3,3],$y->value()->shape());
        $grads = $tape->gradient($y,array_merge([$a,$b],$layer0->weights()));
        $this->assertCount(4,$grads);
        $this->assertEquals([3,1],$grads[0]->shape());
        $this->assertEquals([3,1],$grads[1]->shape());
        $this->assertEquals([1,3],$grads[2]->shape());
        $this->assertEquals([3],$grads[3]->shape());

        $this->assertEquals('[[4,4,4],[7,7,7],[6,6,6]]',$K->toString($y->value()));
        $this->assertEquals('[[3],[3],[3]]',$K->toString($grads[0]));
        $this->assertEquals('[[3],[3],[3]]',$K->toString($grads[1]));
        $this->assertEquals('[[17,17,17]]',$K->toString($grads[2]));
        $this->assertEquals('[3,3,3]',$K->toString($grads[3]));
    }

    public function testLayerInNestGraphNormal()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $layer0 = $nn->layers->Dense(3, kernel_initializer:'ones');
        $func0 = $g->Function(function($a,$b) use ($K,$g,$layer0) {
            $x = $g->add($a,$b);
            $y = $layer0($x,true);
            return $y;
        });
        $func1 = $g->Function(function($a,$b) use ($K,$g,$func0) {
            $y = $func0($a,$b);
            return $y;
        });

        $a = $g->Variable($K->array([[2],[3]]));
        $b = $g->Variable($K->array([[2],[4]]));

        // building
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($func1,$a,$b) {
                $y = $func1($a,$b);
                return $y;
            }
        );
        $this->assertEquals([2,3],$y->value()->shape());
        $grads = $tape->gradient($y,array_merge([$a,$b],$layer0->weights()));
        $this->assertCount(4,$grads);
        $this->assertEquals([2,1],$grads[0]->shape());
        $this->assertEquals([2,1],$grads[1]->shape());
        $this->assertEquals([1,3],$grads[2]->shape());
        $this->assertEquals([3],$grads[3]->shape());

        $this->assertEquals('[[4,4,4],[7,7,7]]',$K->toString($y->value()));
        $this->assertEquals('[[3],[3]]',$K->toString($grads[0]));
        $this->assertEquals('[[3],[3]]',$K->toString($grads[1]));
        $this->assertEquals('[[11,11,11]]',$K->toString($grads[2]));
        $this->assertEquals('[2,2,2]',$K->toString($grads[3]));

        $a = $g->Variable($K->array([[2],[3],[3]]));
        $b = $g->Variable($K->array([[2],[4],[3]]));
        // built graph
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($func1,$a,$b) {
                $y = $func1($a,$b);
                return $y;
            }
        );
        $this->assertEquals([3,3],$y->value()->shape());
        $grads = $tape->gradient($y,array_merge([$a,$b],$layer0->weights()));
        $this->assertCount(4,$grads);
        $this->assertEquals([3,1],$grads[0]->shape());
        $this->assertEquals([3,1],$grads[1]->shape());

        $this->assertEquals('[[4,4,4],[7,7,7],[6,6,6]]',$K->toString($y->value()));
        $this->assertEquals('[[3],[3],[3]]',$K->toString($grads[0]));
        $this->assertEquals('[[3],[3],[3]]',$K->toString($grads[1]));
        $this->assertEquals('[[17,17,17]]',$K->toString($grads[2]));
        $this->assertEquals('[3,3,3]',$K->toString($grads[3]));
    }

    public function testLossWithGraphNormal()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $loss0 = $nn->losses->SparseCategoricalCrossEntropy(from_logits:true);
        $func0 = $g->Function(function($x,$z) use ($K,$g) {
            $y = $g->add($x,$z);
            return $y;
        });

        $t = $g->Variable($K->array([1, 2],NDArray::int32),trainable:false);
        $x = $g->Variable($K->array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]]));
        $z = $g->Variable($K->zerosLike($x->value()));

        // building
        $outputs = $nn->with($tape=$g->GradientTape(),
            function() use ($K,$func0,$loss0,$t,$x,$z) {
                $x = $func0($x,$z);
                $outputs = $loss0($t,$x);
                return $outputs;
            }
        );
        $gradients = $tape->gradient($outputs, $x);
        $this->assertStringStartsWith("0.98689",$mo->toString($outputs->value()));
        $this->assertEquals([2,3],$gradients->shape());
        $this->assertTrue($mo->la()->isclose(
            $mo->array([[0.11335728, -0.22118607,  0.10782879],
                        [ 0.12457169,  0.25085658, -0.3754283 ]]),
            $K->ndarray($gradients)));

        // built graph
        $outputs = $nn->with($tape=$g->GradientTape(),
            function() use ($func0,$loss0,$t,$x,$z) {
                $x = $func0($x,$z);
                $outputs = $loss0($t,$x);
                return $outputs;
            }
        );
        $gradients = $tape->gradient($outputs, $x);
        $this->assertStringStartsWith("0.98689",$mo->toString($outputs->value()));
        $this->assertEquals([2,3],$gradients->shape());
        $this->assertTrue($mo->la()->isclose(
            $mo->array([[0.11335728, -0.22118607,  0.10782879],
                        [ 0.12457169,  0.25085658, -0.3754283 ]]),
            $K->ndarray($gradients)));
    }

    public function testLossWithGraphCompile()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $loss0 = $nn->losses->SparseCategoricalCrossEntropy(from_logits:true);
        $func0 = $g->Function(function($x,$z,$t) use ($K,$g,$loss0) {
            $y = $g->add($x,$z);
            $outputs = $loss0($t,$y);
            return $outputs;
        });

        $t = $g->Variable($K->array([1, 2],NDArray::int32));
        $x = $g->Variable($K->array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]]));
        $z = $g->Variable($K->zerosLike($x->value()));

        // building
        $outputs = $nn->with($tape=$g->GradientTape(),
            function() use ($K,$func0,$loss0,$t,$x,$z) {
                $outputs = $func0($x,$z,$t);
                return $outputs;
            }
        );
        $gradients = $tape->gradient($outputs, $x);
        $this->assertStringStartsWith("0.98689",$mo->toString($outputs->value()));
        $this->assertEquals([2,3],$gradients->shape());
        $this->assertTrue($mo->la()->isclose(
            $mo->array([[0.11335728, -0.22118607,  0.10782879],
                        [ 0.12457169,  0.25085658, -0.3754283 ]]),
            $K->ndarray($gradients)));

        // built graph
        $outputs = $nn->with($tape=$g->GradientTape(),
            function() use ($func0,$loss0,$t,$x,$z) {
                $outputs = $func0($x,$z,$t);
                return $outputs;
            }
        );
        $gradients = $tape->gradient($outputs, $x);
        $this->assertStringStartsWith("0.98689",$mo->toString($outputs->value()));
        $this->assertEquals([2,3],$gradients->shape());
        $this->assertTrue($mo->la()->isclose(
            $mo->array([[0.11335728, -0.22118607,  0.10782879],
                        [ 0.12457169,  0.25085658, -0.3754283 ]]),
            $K->ndarray($gradients)));
    }

    public function testBackwardDiscardedVariableInBatchBoundary1()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();
        $rnn = $nn->layers()->GRU(2,
            return_state:true, return_sequences:true,
            recurrent_initializer:'glorot_uniform'
        );

        $func0 = $g->Function(function($x,$state) use ($K,$g,$rnn) {
            [$y,$dmyStates] = $rnn->forward($x,true,[$state]);
            return $y;
        });

        $x = $g->Variable($K->ones([10,4,2]));
        $state = $g->Variable($K->zeros([10,2]));

        // building
        $outputs = $nn->with($tape=$g->GradientTape(),
            function() use ($K,$func0,$x,$state) {
                $outputs = $func0($x,$state);
                return $outputs;
            }
        );
        $params = $rnn->trainableVariables();
        $gradients = $tape->gradient($outputs, array_merge([$x],$params));
        $this->assertEquals([10,4,2],$outputs->shape());
        $this->assertCount(4,$gradients);
        $this->assertEquals([10,4,2],$gradients[0]->shape()); // dx

        // built graph
        $x = $g->Variable($K->ones([2,4,2]));
        $state = $g->Variable($K->zeros([2,2]));

        $outputs = $nn->with($tape=$g->GradientTape(),
            function() use ($K,$func0,$x,$state) {
                $outputs = $func0($x,$state);
                return $outputs;
            }
        );
        $params = $rnn->trainableVariables();
        $gradients = $tape->gradient($outputs, array_merge([$x],$params));
        $this->assertEquals([2,4,2],$outputs->shape());
        $this->assertCount(4,$gradients);
        $this->assertEquals([2,4,2],$gradients[0]->shape());
    }

    public function testBackwardDiscardedVariableInBatchBoundary2()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $func0 = $g->Function(function($x1,$x2) use ($K,$g) {
            $y1 = $g->add($x1,$x2);
            $y2 = $g->add($y1,$x1);
            $y3 = $g->add($y1,$y2);
            return $y2;
        });

        // building
        $x1 = $g->Variable($K->ones([10,2]));
        $x2 = $g->Variable($K->ones([10,2]));

        $outputs = $nn->with($tape=$g->GradientTape(),
            function() use ($K,$func0,$x1,$x2) {
                $outputs = $func0($x1,$x2);
                return $outputs;
            }
        );
        $gradients = $tape->gradient($outputs, [$x1,$x2]);
        $this->assertEquals([10,2],$outputs->shape());
        $this->assertCount(2,$gradients);
        $this->assertEquals([10,2],$gradients[0]->shape()); // dx
        $this->assertEquals([10,2],$gradients[1]->shape()); // dx

        // built graph
        $x1 = $g->Variable($K->ones([2,2]));
        $x2 = $g->Variable($K->ones([2,2]));

        $outputs = $nn->with($tape=$g->GradientTape(),
            function() use ($K,$func0,$x1,$x2) {
                $outputs = $func0($x1,$x2);
                return $outputs;
            }
        );
        $gradients = $tape->gradient($outputs, [$x1,$x2]);
        $this->assertEquals([2,2],$outputs->shape());
        $this->assertCount(2,$gradients);
        $this->assertEquals([2,2],$gradients[0]->shape()); // dx
        $this->assertEquals([2,2],$gradients[1]->shape()); // dx
    }

    public function testBackwardDiscardedVariableInBatchBoundary3()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $func1 = $g->Function(function($x1,$x2) use ($K,$g) {
            $y1 = $g->add($x1,$x2);
            $y2 = $g->add($y1,$x1);
            $y3 = $g->add($y1,$y2);
            return [$y2,$y3];
        });

        $func0 = $g->Function(function($x1,$x2) use ($K,$g,$func1) {
            [$y,$dmy] = $func1($x1,$x2);
            return $y;
        });

        // building
        $x1 = $g->Variable($K->ones([10,2]));
        $x2 = $g->Variable($K->ones([10,2]));

        $outputs = $nn->with($tape=$g->GradientTape(),
            function() use ($K,$func0,$x1,$x2) {
                $outputs = $func0($x1,$x2);
                return $outputs;
            }
        );
        $gradients = $tape->gradient($outputs, [$x1,$x2]);
        $this->assertEquals([10,2],$outputs->shape());
        $this->assertCount(2,$gradients);
        $this->assertEquals([10,2],$gradients[0]->shape()); // dx
        $this->assertEquals([10,2],$gradients[1]->shape()); // dx

        // built graph
        $x1 = $g->Variable($K->ones([2,2]));
        $x2 = $g->Variable($K->ones([2,2]));

        $outputs = $nn->with($tape=$g->GradientTape(),
            function() use ($K,$func0,$x1,$x2) {
                $outputs = $func0($x1,$x2);
                return $outputs;
            }
        );
        $gradients = $tape->gradient($outputs, [$x1,$x2]);
        $this->assertEquals([2,2],$outputs->shape());
        $this->assertCount(2,$gradients);
        $this->assertEquals([2,2],$gradients[0]->shape()); // dx
        $this->assertEquals([2,2],$gradients[1]->shape()); // dx
    }
}