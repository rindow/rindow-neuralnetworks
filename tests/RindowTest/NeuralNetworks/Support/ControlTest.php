<?php
namespace RindowTest\NeuralNetworks\Support\ControlTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Support\Control\Execute;
use Rindow\NeuralNetworks\Support\Control\Context;
use Throwable;

class TestContext implements Context
{
    protected $session;
    protected $lasterror;

    public function session()
    {
        return $this->session;
    }

    public function lasterror()
    {
        return $this->lasterror;
    }

    public function enter() : void
    {
        $this->session = 'opened';
    }

    public function exit(Throwable $e=null) : bool
    {
        $this->session = 'closed';
        if($e) {
            $this->lasterror = $e->getMessage();
        } else {
            $this->lasterror = 'success';
        }
        return false;
    }
}

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

    public function testWithBasic()
    {
        /// success
        $results = new \stdClass();
        Execute::with(new TestContext(),function($context) use ($results) {
            $results->session = $context->session();
            $results->context = $context;
        });
        $this->assertEquals('opened',$results->session);
        $this->assertEquals('closed',$results->context->session());
        $this->assertEquals('success',$results->context->lasterror());

        /// nest
        $results = new \stdClass();
        Execute::with($ctxA=new TestContext(),function($ctxA) use ($results) {
            Execute::with($ctxB=new TestContext(),function($ctxB) use ($results) {
                $results->session = $ctxB->session();
            });
            $this->assertEquals('closed',$ctxB->session());
            $this->assertEquals('success',$ctxB->lasterror());
            $results->session = $ctxA->session();
        });
        $this->assertEquals('opened',$results->session);
        $this->assertEquals('closed',$ctxA->session());
        //$this->assertEquals('closed',$ctxB->session());
        $this->assertEquals('success',$ctxA->lasterror());

        /// success
        $results = new \stdClass();
        Execute::with(new TestContext(),function($context) use ($results) {
            $results->session = $context->session();
            $results->context = $context;
        });
        $this->assertEquals('opened',$results->session);
        $this->assertEquals('closed',$results->context->session());
        $this->assertEquals('success',$results->context->lasterror());
    }

    public function testWithOptional()
    {
        $results = new \stdClass();
        [$context,$a,$b,$opt1,$opt2] = Execute::with($ctx=new TestContext(),
            args:[1,2,'opt2'=>4,'opt1'=>3],
            func:function($context,$a,$b,$opt1='def1',$opt2='def2') use ($results) {
                $results->session = $context->session();
                $results->context = $context;
                return [$context,$a,$b,$opt1,$opt2];
            }
        );
        $this->assertEquals(1,$a);
        $this->assertEquals(2,$b);
        $this->assertEquals(3,$opt1);
        $this->assertEquals(4,$opt2);
        $this->assertTrue($results->context===$ctx);
    }

    public function testWithoutCtx()
    {
        $ctx = 'DUMMY';
        $results = new \stdClass();
        [$a,$b,$opt1,$opt2] = Execute::with($ctx=new TestContext(),
            args:[1,2,'opt2'=>4,'opt1'=>3],
            without_ctx:true,
            func:function($a,$b,$opt1='def1',$opt2='def2') use ($results) {
                return [$a,$b,$opt1,$opt2];
            }
        );
        $this->assertEquals(1,$a);
        $this->assertEquals(2,$b);
        $this->assertEquals(3,$opt1);
        $this->assertEquals(4,$opt2);
        $this->assertInstanceof(Context::class,$ctx);
    }

    public function testBuilder()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        /// success
        $results = new \stdClass();
        $nn->with(new TestContext(),function($context) use ($results) {
            $results->session = $context->session();
            $results->context = $context;
        });
        $this->assertEquals('opened',$results->session);
        $this->assertEquals('closed',$results->context->session());
        $this->assertEquals('success',$results->context->lasterror());

        /// nest
        $results = new \stdClass();
        $nn->with($ctxA=new TestContext(),function($ctxA) use ($nn,$results) {
            $nn->with($ctxB=new TestContext(),function($ctxB) use ($results) {
                $results->session = $ctxB->session();
            });
            $this->assertEquals('closed',$ctxB->session());
            $this->assertEquals('success',$ctxB->lasterror());
            $results->session = $ctxA->session();
        });
        $this->assertEquals('opened',$results->session);
        $this->assertEquals('closed',$ctxA->session());
        //$this->assertEquals('closed',$ctxB->session());
        $this->assertEquals('success',$ctxA->lasterror());

        /// success
        $results = new \stdClass();
        $nn->with(new TestContext(),function($context) use ($results) {
            $results->session = $context->session();
            $results->context = $context;
        });
        $this->assertEquals('opened',$results->session);
        $this->assertEquals('closed',$results->context->session());
        $this->assertEquals('success',$results->context->lasterror());
    }
}
