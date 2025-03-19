<?php
namespace RindowTest\NeuralNetworks\Gradient\Core\GradientTapeTest;

use InvalidArgumentException;
use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Matrix\NDArrayPhp;
use Rindow\Math\Matrix\NDArrayCL;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;
use Rindow\NeuralNetworks\Gradient\Variable;

class TestNDArrayPhp extends NDArrayPhp
{
    public $_debug_name;
}

class TestNDArrayCL extends NDArrayCL
{
    public $_debug_name;
}

class AbstractTestFunction extends AbstractFunction
{
    protected bool $inTraining = false;
    protected ?object $logger;
    protected ?string $name;

    public function __construct(object $backend, ?object $logger=null, ?string $name=null)
    {
        parent::__construct($backend);
        $this->logger = $logger;
        $this->name = $name;
    }

    public function name() : ?string
    {
        return $this->name;
    }

    protected function call(array $inputs) : array
    {
        $args = array_reduce(
            $this->inputsVariables,function($c,$v) {
                return array_merge($c,[$v->name()]);
            },[]);
        $this->logger->log('call '.$this->name.'('.implode(',',$args).')');
        $K = $this->backend;
        if($this->inTraining) {
            throw new \Exception('Illegal forward');
        }

        $outputs = [];
        for($i=0;$i<$this->numOfOutputs;$i++) {
            $outputs[] = $K->onesLike($inputs[0]);
        }
        $this->inTraining = true;
        return $outputs;
    }

    public function checkGetValueType($v)
    {
        $K = $this->backend;
        if($v instanceof NDArray) {
            if($K->scalar($K->sum($v)) == 0) {
                return 'Z';
            } else {
                return 'N';
            }
        } else {
            return '?';
        }
    }

    protected function differentiate(array $dOutputs) : array
    {
        $args = array_reduce(
            $dOutputs,function($c,$v) {
                return array_merge(
                    $c,
                    (property_exists($v,'_debug_name')?[$v->_debug_name]:[
                        $this->checkGetValueType($v)
                    ]));
            },[]);
        $this->logger->log('diff '.$this->name.'('.implode(',',$args).') gen='.$this->generation);
        $K = $this->backend;
        if(!$this->inTraining) {
            throw new \Exception('Illegal backward');
        }

        if(count($dOutputs)!=$this->numOfOutputs) {
            throw new InvalidArgumentException(
                'Num of dOutputs must be '.$this->numOfOutputs.' values. give '.count($dOutputs).' values');
        }
        foreach($dOutputs as $o) {
            if(!($o instanceof NDArray)) {
                $type = is_object($o)?get_class($o):gettype($o);
                throw new InvalidArgumentException('dOutputs must be list of NDArray: '.$type.' given');
            }
        }

        $inputs = $this->inputsVariables;
        $dInputs = [];
        foreach($inputs as $key => $i) {
            $v = $K->onesLike($i->value());
            if($v instanceof NDArrayCL) {
                $v = new TestNDArrayCL($K->queue(),$v->buffer(),$v->dtype(),$v->shape(),$v->offset(),
                    service:$K->localMatrixOperator()->service());
            } else {
                $v = new TestNDArrayPhp($v->buffer(),$v->dtype(),$v->shape(),$v->offset(),
                    service:$K->localMatrixOperator()->service());
            }
            $v->_debug_name = 'dIn'.$key.'@'.$this->name;
            $dInputs[] = $v;
        }
        $this->inTraining = false;
        return $dInputs;
    }

    public function __invoke(mixed ...$inputs) : mixed
    {
        $outputs = parent::__invoke(...$inputs);
        $tmp = $outputs;
        if(!is_array($tmp)) {
            $tmp = [$tmp];
        }
        foreach ($tmp as $key => $v) {
            $v->setName('Out'.$key.'@'.$this->name);
        }
        return $outputs;
    }
}

class TestFunction2In1Out extends AbstractTestFunction
{
    protected int $numOfInputs = 2;
    protected int $numOfOutputs = 1;
}

class TestFunction1In2Out extends AbstractTestFunction
{
    protected int $numOfInputs = 1;
    protected int $numOfOutputs = 2;
}

class TestBuilder
{
    protected $backend;
    protected $logger;
    public function __construct($backend,$logger)
    {
        $this->backend = $backend;
        $this->logger = $logger;
    }

    public function twoInOneOut($x,$y,$logger=null,$name=null)
    {
        $logger = $logger ?? $this->logger;
        $func = new TestFunction2In1Out(
            $this->backend, logger:$logger, name:$name);
        return $func($x,$y);
    }

    public function oneInTwoOut($x,$logger=null,$name=null)
    {
        $logger = $logger ?? $this->logger;
        $func = new TestFunction1In2Out(
            $this->backend, logger:$logger, name:$name);
        return $func($x);
    }
}

class Logger
{
    protected $logtexts = [];
    public function log($message)
    {
        $this->logtexts[] = $message;
    }
    public function getLog()
    {
        return $this->logtexts;
    }
}

class GradientTapeTest extends TestCase
{
    public function newMatrixOperator()
    {
        return new MatrixOperator();
    }

    public function newNeuralNetworks($mo)
    {
        return new NeuralNetworks($mo);
    }

    public function testPureTreeNetwork()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $logger = new Logger();
        $b = new TestBuilder($nn->backend(),$logger);

        $x0 = $g->Variable($mo->array(0.5),name:'x0');
        $x1 = $g->Variable($mo->array(0.5),name:'x1');
        $x2 = $g->Variable($mo->array(0.5),name:'x2');
        $x3 = $g->Variable($mo->array(0.5),name:'x3');
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$b,$x0,$x1,$x2,$x3){
                $y = $b->twoInOneOut(
                    $b->twoInOneOut(
                        $b->twoInOneOut($x0,$x1,name:'F1'),
                        $x2, name:'F2'),
                    $x3, name:'F3');
                return $y;
            }
        );

        $gradients = $tape->gradient($y,$x3);
        // N = NDArray, U = Undetermined
        $this->assertEquals([
            'call F1(x0,x1)',
            'call F2(Out0@F1,x2)',
            'call F3(Out0@F2,x3)',
            'diff F3(N) gen=2',
            'diff F2(dIn0@F3) gen=1',
            'diff F1(dIn0@F2) gen=0',
        ],$logger->getLog());
        $this->assertTrue(true);
    }

    public function testBroadcastNetwork1()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $g = $nn->gradient();
        $logger = new Logger();
        $b = new TestBuilder($nn->backend(),$logger);

        $x0 = $g->Variable($mo->array(0.5),name:'x0');
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$b,$x0){
                $y = $b->twoInOneOut(
                    $b->twoInOneOut(
                        $b->twoInOneOut($x0,$x0,name:'F1'),
                        $x0,name:'F2'),
                    $x0, name:'F3');
                return $y;
            }
        );

        $gradients = $tape->gradient($y,$x0);
        // N = NDArray, U = Undetermined
        $this->assertEquals([
            'call F1(x0,x0)',
            'call F2(Out0@F1,x0)',
            'call F3(Out0@F2,x0)',
            'diff F3(N) gen=2',
            'diff F2(dIn0@F3) gen=1',
            'diff F1(dIn0@F2) gen=0',
        ],$logger->getLog());
        $this->assertTrue(true);
    }

    public function testBroadcastOutputs()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $g = $nn->gradient();
        $logger = new Logger();
        $b = new TestBuilder($nn->backend(),$logger);

        $x0 = $g->Variable($mo->array(0.5),name:'x0');
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$b,$x0){
                $y0 = $b->twoInOneOut($x0,$x0,name:'F1');
                $y1 = $b->twoInOneOut($x0,$y0,name:'F2');
                $y = $b->twoInOneOut($y0,$y1,name:'F3');
                return $y;
            }
        );

        $gradients = $tape->gradient($y,$x0);
        // N = NDArray, U = Undetermined
        $this->assertEquals([
            'call F1(x0,x0)',
            'call F2(x0,Out0@F1)',
            'call F3(Out0@F1,Out0@F2)',
            'diff F3(N) gen=2',
            'diff F2(dIn1@F3) gen=1',
            'diff F1(N) gen=0', // <=  Add dIn0@F3,dIn1@F2
        ],$logger->getLog());
        $this->assertTrue(true);
    }

    public function testForkNetwork()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $g = $nn->gradient();
        $logger = new Logger();
        $b = new TestBuilder($nn->backend(),$logger);

        $x0 = $g->Variable($mo->array(0.5),name:'x0');
        $x1 = $g->Variable($mo->array(0.5),name:'x1');
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$b,$x0,$x1){
                $y0 = $b->twoInOneOut($x0,$x1,name:'F1');
                [$y1,$y2] = $b->oneInTwoOut($y0,name:'F2');
                $y3 = $b->twoInOneOut($y1,$x0,name:'F3');
                $y = $b->twoInOneOut($y2,$y3,name:'F4');
                return $y;
            }
        );

        $gradients = $tape->gradient($y,$x0);
        // N = NDArray, U = Undetermined
        $this->assertEquals([
            'call F1(x0,x1)',
            'call F2(Out0@F1)',
            'call F3(Out0@F2,x0)',
            'call F4(Out1@F2,Out0@F3)',
            'diff F4(N) gen=3',
            'diff F3(dIn1@F4) gen=2',
            'diff F2(dIn0@F3,dIn0@F4) gen=1',
            'diff F1(dIn0@F2) gen=0',
        ],$logger->getLog());
        $this->assertTrue(true);
    }

    public function testOpenedForkNetwork1()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $g = $nn->gradient();
        $logger = new Logger();
        $b = new TestBuilder($nn->backend(),$logger);

        $x0 = $g->Variable($mo->array(0.5),name:'x0');
        $x1 = $g->Variable($mo->array(0.5),name:'x1');
        $y = $nn->with($tape=$g->GradientTape(),
            function() use ($g,$b,$x0,$x1){
                [$y0,$y1] = $b->oneInTwoOut($x0,name:'F1');
                $y = $b->twoInOneOut($x1,$y0,name:'F2');
                // $y1 is not used
                return $y;
            }
        );

        $gradients = $tape->gradient($y,$x0);
        // N = NDArray, U = Undetermined, UN = UndeterminedNDArray, Z = Zero
        $this->assertEquals([
            'call F1(x0)',
            'call F2(x1,Out0@F1)',
            'diff F2(N) gen=1',
            'diff F1(dIn1@F2,Z) gen=0',
        ],$logger->getLog());
        $this->assertTrue(true);
    }
}
