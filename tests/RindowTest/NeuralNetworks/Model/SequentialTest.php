<?php
namespace RindowTest\NeuralNetworks\Model\SequentialTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Model\AbstractModel;
use Rindow\NeuralNetworks\Layer\Dense;
use Rindow\NeuralNetworks\Callback\AbstractCallback;
use Rindow\NeuralNetworks\Data\Dataset\DatasetFilter;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Layer\Layer;
use Rindow\Math\Plot\Plot;
use Rindow\Math\Plot\Renderer\GDDriver;
use Interop\Polite\Math\Matrix\NDArray;

class WeightLog extends AbstractCallback
{
    protected $mo;
    protected $prev_w;
    protected $gradlog = [];
    public function __construct($mo)
    {
        parent::__construct();
        $this->mo = $mo;
        $this->prev_w = null;
    }

    public function onEpochEnd(int $epoch, array $metrics=null) : void
    {
        $model = $this->getModel();
        $K = $model->backend();
        $weights = array_map(fn($w)=>$w->value(),$model->trainableVariables());
        if($this->prev_w==null) {
            $this->prev_w = $weights;
            return;
        }
        $num = 0;
        $next = [];
        foreach($weights as $key => $w) {
            $prev = $this->prev_w[$key];
            $w = $K->ndarray($w);
            $prev = $K->ndarray($prev);
            $g = $this->mo->op($prev,'-',$w);
            if(in_array($num,[100])) {
                for($i=0;$i<3;$i++) {
                    $name = 'g'.$num.$i.'('.implode(',',$w->shape()).')';
                    if(!isset($this->gradlog[$name])) {
                        $this->gradlog[$name] = [];
                    }
                    $gg = $this->mo->la()->slice($w,[0,$i*128],[-1,128]);
                    $this->gradlog[$name][] = abs($this->mo->amax($gg));
                }
            } else {
                $name = 'g'.$num.'('.implode(',',$w->shape()).')';
                if(!isset($this->gradlog[$name])) {
                    $this->gradlog[$name] = [];
                }
                $this->gradlog[$name][] = abs($this->mo->amax($g));
            }
            $num++;
            $next[] = $this->mo->copy($w);
        }
        $this->prev_w = $next;
    }

    public function getGradlog()
    {
        return $this->gradlog;
    }
}

class TestFilter implements DatasetFilter
{
    protected $mo;
    public function __construct($mo = null)
    {
        $this->mo = $mo;
    }
    public function translate(
        iterable $inputs, iterable $tests=null, $options=null) : array
    {
        $batchSize= count($inputs);
        $cols = count($inputs[0])-1;
        $inputsNDArray = $this->mo->la()->alloc([$batchSize,$cols]);
        $testsNDArray = $this->mo->la()->alloc([$batchSize,1],dtype:NDArray::int32);
        foreach ($inputs as $i => $row) {
            $testsNDArray[$i][0] = (float)array_pop($row);
            for($j=0;$j<$cols;$j++) {
                $inputsNDArray[$i][$j] = (float)$row[$j];
            }
        }
        return [$inputsNDArray,$testsNDArray->reshape([count($testsNDArray)])];
    }
}

class TestCustomModel extends AbstractModel
{
    protected $seq;
    public function __construct($builder,$seq)
    {
        parent::__construct($builder);
        $this->seq = $seq;
    }

    public function call($inputs)
    {
        $outputs = $this->seq->forward($inputs);
        return $outputs;
    }
}

class TestCustomSubModel extends AbstractModel
{
    protected Variable $param1;
    protected Layer $sublayer;

    public function __construct($builder)
    {
        parent::__construct($builder);
        $g = $builder->gradient();
        $this->param1 = $g->Variable([0,0]);
        $this->sublayer = $builder->layers->Dense(10);
    }

    public function call($inputs,$training=null)
    {
        $g = $this->builder->gradient();
        $outputs = $g->square($inputs);

        // dummy
        $this->sublayer->forward($inputs);

        return $outputs;
    }
}

class SequentialTest extends TestCase
{
    protected $plot = false;

    public function newMatrixOperator()
    {
        return new MatrixOperator();
    }

    public function newNeuralNetworks($mo)
    {
        return new NeuralNetworks($mo);
    }

    public function getPlotConfig()
    {
        return [
            'renderer.skipCleaning' => true,
            'renderer.skipRunViewer' => getenv('PLOT_RENDERER_SKIP') ? true : false,
        ];
    }

    public function setUp() : void
    {
        $this->plot = true;
    }

    public function testCleanUp()
    {
        $renderer = new GDDriver();
        $renderer->cleanUp();
        $this->assertTrue(true);
    }

    public function testComplieDefaults()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense(
                $units=4,
                input_shape:[2],
                activation:'sigmoid'),
            $nn->layers()->Dense(
                $units=3,
                activation:'softmax'),
        ]);

        $model->compile();
        $inputsVariable = $g->Variable($K->zeros([1,2]));
        $outputsVariable = $nn->with($tape=$g->GradientTape(),
            function() use ($model,$inputsVariable) {
                $outputsVariable = $model->forward($inputsVariable,true);
                return $outputsVariable;
            }
        );
        $layers = $model->layers();
        $lossFunction = $model->lossFunction();
        $weights = $model->trainableVariables();
        $grads = $tape->gradient($outputsVariable,$weights);

        $this->assertCount(2,$layers);
        $this->assertInstanceof(
            'Rindow\NeuralNetworks\Layer\Dense',$layers[0]);
        $this->assertInstanceof(
            'Rindow\NeuralNetworks\Activation\Sigmoid',$layers[0]->getActivation());
        $this->assertInstanceof(
            'Rindow\NeuralNetworks\Layer\Dense',$layers[1]);
        $this->assertInstanceof(
            'Rindow\NeuralNetworks\Activation\Softmax',$layers[1]->getActivation());
        $this->assertInstanceof(
            'Rindow\NeuralNetworks\Loss\SparseCategoricalCrossEntropy',$lossFunction);
        $this->assertInstanceof(
            \Rindow\NeuralNetworks\Activation\Softmax::class,
            $model->layers()[1]->getActivation());
        $this->assertInstanceof(
            \Rindow\NeuralNetworks\Loss\SparseCategoricalCrossEntropy::class,
            $model->lossFunction());

        $this->assertCount(4,$weights);
        $this->assertInstanceof('Interop\Polite\Math\Matrix\NDArray',$weights[0]);
        $this->assertInstanceof('Interop\Polite\Math\Matrix\NDArray',$weights[1]);
        $this->assertInstanceof('Interop\Polite\Math\Matrix\NDArray',$weights[2]);
        $this->assertInstanceof('Interop\Polite\Math\Matrix\NDArray',$weights[3]);

        $this->assertEquals([2,4], $weights[0]->shape());
        $this->assertEquals([4],   $weights[1]->shape());
        $this->assertEquals([4,3], $weights[2]->shape());
        $this->assertEquals([3],   $weights[3]->shape());

        $this->assertCount(4,$grads);
        $this->assertInstanceof('Interop\Polite\Math\Matrix\NDArray',$grads[0]);
        $this->assertInstanceof('Interop\Polite\Math\Matrix\NDArray',$grads[1]);
        $this->assertInstanceof('Interop\Polite\Math\Matrix\NDArray',$grads[2]);
        $this->assertInstanceof('Interop\Polite\Math\Matrix\NDArray',$grads[3]);

        $this->assertEquals([2,4],$grads[0]->shape());
        $this->assertEquals([4],  $grads[1]->shape());
        $this->assertEquals([4,3],$grads[2]->shape());
        $this->assertEquals([3],  $grads[3]->shape());

        $this->assertNotEquals(spl_object_hash($weights[0]),spl_object_hash($grads[0]));
        $this->assertNotEquals(spl_object_hash($weights[1]),spl_object_hash($grads[1]));
        $this->assertNotEquals(spl_object_hash($weights[2]),spl_object_hash($grads[2]));
        $this->assertNotEquals(spl_object_hash($weights[3]),spl_object_hash($grads[3]));

        //$this->assertNotEquals($mo->zeros([2,4])->toArray(), $weights[0]->toArray());
        //$this->assertEquals(   $mo->zeros([4])->toArray(),   $weights[1]->toArray());
        //$this->assertNotEquals($mo->zeros([4,3])->toArray(), $weights[2]->toArray());
        //$this->assertEquals(   $mo->zeros([3])->toArray(),   $weights[3]->toArray());
        //
        //$this->assertEquals($mo->zeros([2,4])->toArray(),    $grads[0]->toArray());
        //$this->assertEquals($mo->zeros([4])->toArray(),      $grads[1]->toArray());
        //$this->assertEquals($mo->zeros([4,3])->toArray(),    $grads[2]->toArray());
        //$this->assertEquals($mo->zeros([3])->toArray(),      $grads[3]->toArray());
    }

    public function testSummary()
    {
        Dense::$nameNumbering = 0;
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,input_shape:[2],
                    activation:'sigmoid'),
            $nn->layers()->Dense($units=2,activation:'softmax'),
        ]);

        //$model->compile();
        ob_start();
        $model->summary();
        $dump = ob_get_clean();
        $display = 
        'Layer(type)                  Output Shape               Param #   '."\n".
        '=================================================================='."\n".
        'dense(Dense)                 (128)                      384       '."\n".
        'dense_1(Dense)               (2)                        258       '."\n".
        '=================================================================='."\n".
        'Total params: 642'."\n";

        $this->assertEquals($display,$dump);
    }

    public function testFitAndPredictWithDefaults()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,input_shape:[2],
                    activation:'sigmoid'),
            $nn->layers()->Dense($units=2,activation:'softmax'),
        ]);

        $model->compile();
        $this->assertInstanceof(
            \Rindow\NeuralNetworks\Activation\Softmax::class,
            $model->layers()[1]->getActivation());
        $this->assertInstanceof(
            \Rindow\NeuralNetworks\Loss\SparseCategoricalCrossEntropy::class,
            $model->lossFunction());

        // training greater or less
        $x = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $t = $mo->array([0, 0, 0, 1, 1, 1],dtype:NDArray::int32);
        $history = $model->fit($x,$t,epochs:100,verbose:0);

        $y = $model->predict($x);
        $this->assertEquals($t->toArray(),$mo->argMax($y,axis:1)->toArray());

        //$plt->plot($mo->array($history['loss']));
        //$plt->plot($mo->array($history['accuracy']));
        //$plt->title('fit and predict');
        //$plt->show();
    }

    public function testFitAndPredictWithNDArrayDataset()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,input_shape:[2],
                    activation:'sigmoid'),
            $nn->layers()->Dense($units=2,activation:'softmax'),
        ]);

        $model->compile();
        $this->assertInstanceof(
            \Rindow\NeuralNetworks\Activation\Softmax::class,
            $model->layers()[1]->getActivation());
        $this->assertInstanceof(
            \Rindow\NeuralNetworks\Loss\SparseCategoricalCrossEntropy::class,
            $model->lossFunction());

        // training greater or less
        $x = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $t = $mo->array([0, 0, 0, 1, 1, 1],dtype:NDArray::int32);
        $dataset = $nn->data->NDArrayDataset($x,
            tests:$t,
            batch_size:64,
            shuffle:true,
        );
        $history = $model->fit($dataset,null,epochs:100,verbose:0);

        $y = $model->predict($x);
        $this->assertEquals($t->toArray(),$mo->argMax($y,axis:1)->toArray());

        //$plt->plot($mo->array($history['loss']));
        //$plt->plot($mo->array($history['accuracy']));
        //$plt->title('fit and predict');
        //$plt->show();
    }

    public function testFitAndPredictWithNCSVDataset()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,input_shape:[2],
                    activation:'sigmoid'),
            $nn->layers()->Dense($units=2,activation:'softmax'),
        ]);

        $model->compile();
        $this->assertInstanceof(
            \Rindow\NeuralNetworks\Activation\Softmax::class,
            $model->layers()[1]->getActivation());
        $this->assertInstanceof(
            \Rindow\NeuralNetworks\Loss\SparseCategoricalCrossEntropy::class,
            $model->lossFunction());

        // training greater or less
        //$x = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        //$t = $mo->array([0, 0, 0, 1, 1, 1]);
        $dataset = $nn->data->CSVDataset(
            __DIR__.'/csv',
            pattern:'@.*\\.csv@',
            batch_size:64,
            shuffle:true,
            filter:new TestFilter($mo),
        );
        $history = $model->fit($dataset,null, epochs: 100, verbose: 0);

        $x = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $t = $mo->array([0, 0, 0, 1, 1, 1],dtype:NDArray::int32);
        $y = $model->predict($x);
        $this->assertEquals($t->toArray(),$mo->argMax($y,axis:1)->toArray());

        //$plt->plot($mo->array($history['loss']));
        //$plt->plot($mo->array($history['accuracy']));
        //$plt->title('fit and predict');
        //$plt->show();
    }

    public function testEvaluateDefault()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,input_shape:[2],
                activation:'sigmoid'),
            $nn->layers()->Dense($units=2,activation:'softmax'),
        ]);

        $model->compile();

        // training greater or less
        $x = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $t = $mo->array([0, 0, 0, 1, 1, 1],dtype:NDArray::int32);
        $history = $model->fit($x,$t,epochs:100,verbose:0);

        $logs = $model->evaluate($x,$t);
        $this->assertLessThan(1.0,$logs['loss']);
        $this->assertEquals(1.0,$logs['accuracy']);
    }

    public function testEvaluateNDArrayDataset()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,input_shape:[2],
                activation:'sigmoid'),
            $nn->layers()->Dense($units=2,activation:'softmax'),
        ]);

        $model->compile();

        // training greater or less
        $x = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $t = $mo->array([0, 0, 0, 1, 1, 1],dtype:NDArray::int32);
        $history = $model->fit($x,$t,epochs:100,verbose:0);

        $dataset = $nn->data->NDArrayDataset($x,
            tests:$t,
            batch_size:64,
            shuffle:true,
        );
        $logs = $model->evaluate($dataset);
        $this->assertLessThan(1.0,$logs['loss']);
        $this->assertEquals(1.0,$logs['accuracy']);
    }

    public function testEvaluateCSVDataset()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,input_shape:[2],
                activation:'sigmoid'),
            $nn->layers()->Dense($units=2,activation:'softmax'),
        ]);

        $model->compile();

        // training greater or less
        $x = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $t = $mo->array([0, 0, 0, 1, 1, 1],dtype:NDArray::int32);
        $history = $model->fit($x,$t,epochs:100,verbose:0);

        $dataset = $nn->data->CSVDataset(
            __DIR__.'/csv',
            pattern:'@.*\\.csv@',
            batch_size:64,
            shuffle:true,
            filter:new TestFilter($mo),
        );
        $logs = $model->evaluate($dataset);
        $this->assertLessThan(1.0,$logs['loss']);
        $this->assertEquals(1.0,$logs['accuracy']);
    }

    public function testFitWithEvaluate()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,input_shape:[2],
                activation:'sigmoid'),
            $nn->layers()->Dense($units=2,
                activation:'softmax'),
        ]);

        $model->compile();

        // training greater or less
        $x = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $t = $mo->array([0, 0, 0, 1, 1, 1],dtype:NDArray::int32);
        $v_x = $mo->array([[5, 1], [1, 5], [2, 6], [6, 1], [1, 7], [7, 2]]);
        $v_t = $mo->array([1, 0, 0, 1, 0, 1],dtype:NDArray::int32);
        $history = $model->fit($x,$t,epochs:100,validation_data:[$v_x,$v_t],verbose:0);

        $this->assertEquals(['loss','accuracy','val_loss','val_accuracy'],array_keys($history));

        if($this->plot) {
            $plt->plot($mo->array($history['loss']),null,null,'loss');
            $plt->plot($mo->array($history['val_loss']),null,null,'val_loss');
            $plt->plot($mo->array($history['accuracy']),null,null,'accuracy');
            $plt->plot($mo->array($history['val_accuracy']),null,null,'val_accuracy');
            $plt->legend();
            $plt->title('Normal with evaluate');
            $plt->show();
        }
    }

    public function testFitAndPredictWithReLUAndAdam()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,
                input_shape:[2],
                kernel_initializer:'he_normal',
                activation:'relu'),
            $nn->layers()->Dense($units=2,
                activation:'softmax'),
        ]);

        $model->compile(
            optimizer:$nn->optimizers()->Adam()
        );
        $this->assertInstanceof(
            \Rindow\NeuralNetworks\Activation\Softmax::class,
            $model->layers()[1]->getActivation());
        $this->assertInstanceof(
            \Rindow\NeuralNetworks\Loss\SparseCategoricalCrossEntropy::class,
            $model->lossFunction());

        // training greater or less
        $x = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $t = $mo->array([0, 0, 0, 1, 1, 1],dtype:NDArray::int32);
        $v_x = $mo->array([[5, 1], [1, 5], [2, 6], [6, 1], [1, 7], [7, 2]]);
        $v_t = $mo->array([1, 0, 0, 1, 0, 1],dtype:NDArray::int32);
        $history = $model->fit($x,$t,epochs:100,validation_data:[$v_x,$v_t],verbose:0);

        $y = $model->predict($x);
        $this->assertEquals($t->toArray(),$mo->argMax($y,axis:1)->toArray());

        if($this->plot) {
            $plt->plot($mo->array($history['loss']),null,null,'loss');
            $plt->plot($mo->array($history['val_loss']),null,null,'val_loss');
            $plt->plot($mo->array($history['accuracy']),null,null,'accuracy');
            $plt->plot($mo->array($history['val_accuracy']),null,null,'val_accuracy');
            $plt->legend();
            $plt->title('ReLU & Adam');
            $plt->show();
        }
    }

    public function testFitWithMeanSquareError()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,input_shape:[2],
                activation:'sigmoid'),
            $nn->layers()->Dense($units=2,
                activation:'softmax'),
        ]);

        $model->compile(
            loss:$nn->losses()->MeanSquaredError()
        );

        // training greater or less

        // training greater or less
        $x = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $t = $mo->array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]]);
        $v_x = $mo->array([[5, 1], [1, 5], [2, 6], [6, 1], [1, 7], [7, 2]]);
        $v_t = $mo->array([[0, 1], [1, 0], [1, 0], [0, 1], [1, 0], [0, 1]]);
        $history = $model->fit($x,$t,epochs:100,validation_data:[$v_x,$v_t],verbose:0);

        $y = $model->predict($x);
        $this->assertEquals($mo->argMax($t,axis:1)->toArray(),
                            $mo->argMax($y,axis:1)->toArray());

        $this->assertEquals(['loss','accuracy','val_loss','val_accuracy'],array_keys($history));

        if($this->plot) {
            $plt->plot($mo->array($history['loss']),null,null,'loss');
            $plt->plot($mo->array($history['val_loss']),null,null,'val_loss');
            $plt->plot($mo->array($history['accuracy']),null,null,'accuracy');
            $plt->plot($mo->array($history['val_accuracy']),null,null,'val_accuracy');
            $plt->legend();
            $plt->title('MeanSquareError');
            $plt->show();
        }
    }

    public function testFitWithDropout()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128, input_shape:[2],
                activation:'relu'),
            $nn->layers()->Dropout($rate=0.15),
            $nn->layers()->Dense($units=2, activation:'softmax'),
        ]);

        $model->compile();

        // training greater or less
        $x = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $t = $mo->array([0, 0, 0, 1, 1, 1],dtype:NDArray::int32);
        $v_x = $mo->array([[5, 1], [1, 5], [2, 6], [6, 1], [1, 7], [7, 2]]);
        $v_t = $mo->array([1, 0, 0, 1, 0, 1],dtype:NDArray::int32);
        $history = $model->fit($x,$t,epochs:100,validation_data:[$v_x,$v_t],verbose:0);

        $this->assertEquals(['loss','accuracy','val_loss','val_accuracy'],array_keys($history));

        if($this->plot) {
            $plt->plot($mo->array($history['loss']),null,null,'loss');
            $plt->plot($mo->array($history['val_loss']),null,null,'val_loss');
            $plt->plot($mo->array($history['accuracy']),null,null,'accuracy');
            $plt->plot($mo->array($history['val_accuracy']),null,null,'val_accuracy');
            $plt->legend();
            $plt->title('Dropout');
            $plt->show();
        }
    }

    public function testFitWithBatchNormalization()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128, input_shape:[2],
                activation:'relu'),
            $nn->layers()->BatchNormalization(),
            $nn->layers()->Dense($units=2, activation:'softmax'),
        ]);

        $model->compile();

        // training greater or less
        $x = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $t = $mo->array([0, 0, 0, 1, 1, 1],dtype:NDArray::int32);
        $v_x = $mo->array([[5, 1], [1, 5], [2, 6], [6, 1], [1, 7], [7, 2]]);
        $v_t = $mo->array([1, 0, 0, 1, 0, 1],dtype:NDArray::int32);
        $history = $model->fit($x,$t,epochs:100,validation_data:[$v_x,$v_t],verbose:0);

        $this->assertEquals(['loss','accuracy','val_loss','val_accuracy'],array_keys($history));

        if($this->plot) {
            $plt->plot($mo->array($history['loss']),null,null,'loss');
            $plt->plot($mo->array($history['val_loss']),null,null,'val_loss');
            $plt->plot($mo->array($history['accuracy']),null,null,'accuracy');
            $plt->plot($mo->array($history['val_accuracy']),null,null,'val_accuracy');
            $plt->legend();
            $plt->title('BatchNormalization');
            $plt->show();
        }
    }

    public function testFitWithBatchNorm()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128, input_shape:[2]),
            $nn->layers()->BatchNormalization(),
            $nn->layers()->Activation('sigmoid'),
            $nn->layers()->Dense($units=2, activation:'softmax'),
        ]);

        $model->compile();

        // training greater or less
        $x = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $t = $mo->array([0, 0, 0, 1, 1, 1],dtype:NDArray::int32);
        $v_x = $mo->array([[5, 1], [1, 5], [2, 6], [6, 1], [1, 7], [7, 2]]);
        $v_t = $mo->array([1, 0, 0, 1, 0, 1],dtype:NDArray::int32);
        $history = $model->fit($x,$t,epochs:100,validation_data:[$v_x,$v_t],verbose:0);

        $this->assertEquals(['loss','accuracy','val_loss','val_accuracy'],array_keys($history));

        if($this->plot) {
            $plt->plot($mo->array($history['loss']),null,null,'loss');
            $plt->plot($mo->array($history['val_loss']),null,null,'val_loss');
            $plt->plot($mo->array($history['accuracy']),null,null,'accuracy');
            $plt->plot($mo->array($history['val_accuracy']),null,null,'val_accuracy');
            $plt->legend();
            $plt->title('BatchNormalization');
            $plt->show();
        }
    }

    public function testFitBinaryClassification()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128, input_shape:[2],
                activation:'sigmoid'),
            $nn->layers()->Dense($units=1,
                activation:'sigmoid'),
        ]);

        $model->compile(
            loss:$nn->losses()->BinaryCrossEntropy(),
            #optimizer:'adam',
        );
        //$model->summary();
        $this->assertInstanceof(
            \Rindow\NeuralNetworks\Activation\Sigmoid::class,
            $model->layers()[1]->getActivation());
        $this->assertInstanceof(
            \Rindow\NeuralNetworks\Loss\BinaryCrossEntropy::class,
            $model->lossFunction());

        // training greater or less
        $x = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $t = $mo->array([0, 0, 0, 1, 1, 1]);
        $v_x = $mo->array([[5, 1], [1, 5], [2, 6], [6, 1], [1, 7], [7, 2]]);
        $v_t = $mo->array([1, 0, 0, 1, 0, 1]);
        $history = $model->fit($x,$t,epochs:100,batch_size:16,validation_data:[$v_x,$v_t],verbose:0);

        $this->assertEquals(['loss','accuracy','val_loss','val_accuracy'],array_keys($history));

        if($this->plot) {
            $plt->plot($mo->array($history['loss']),null,null,'loss');
            $plt->plot($mo->array($history['val_loss']),null,null,'val_loss');
            $plt->plot($mo->array($history['accuracy']),null,null,'accuracy');
            $plt->plot($mo->array($history['val_accuracy']),null,null,'val_accuracy');
            $plt->legend();
            $plt->title('BinaryClassification');
            $plt->show();
        }
    }

    public function testFitOnehotCategoricalClassification()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,input_shape:[2],
                activation:'sigmoid'),
            $nn->layers()->Dense($units=2,
                activation:'softmax'),
        ]);

        $model->compile(
            loss:$nn->losses()->CategoricalCrossEntropy(),
        );
        $this->assertInstanceof(
            \Rindow\NeuralNetworks\Activation\Softmax::class,
            $model->layers()[1]->getActivation());
        $this->assertInstanceof(
            \Rindow\NeuralNetworks\Loss\CategoricalCrossEntropy::class,
            $model->lossFunction());

        // training greater or less
        $x = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $t = $mo->array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]]);
        $v_x = $mo->array([[5, 1], [1, 5], [2, 6], [6, 1], [1, 7], [7, 2]]);
        $v_t = $mo->array([[0, 1], [1, 0], [1, 0], [0, 1], [1, 0], [0, 1]]);
        $history = $model->fit($x,$t,epochs:100,validation_data:[$v_x,$v_t],verbose:0);

        $this->assertEquals(['loss','accuracy','val_loss','val_accuracy'],array_keys($history));

        if($this->plot) {
            $plt->plot($mo->array($history['loss']),null,null,'loss');
            $plt->plot($mo->array($history['val_loss']),null,null,'val_loss');
            $plt->plot($mo->array($history['accuracy']),null,null,'accuracy');
            $plt->plot($mo->array($history['val_accuracy']),null,null,'val_accuracy');
            $plt->legend();
            $plt->title('CategoricalCrossEntropy');
            $plt->show();
        }
    }

    public function testFitConv1DandMaxPooling1D()
    {
        $mo = $this->newMatrixOperator();
        if($mo->isAdvanced()) {
            $num_of_filters=128;
            $epoch = 300;
        } else {
            $num_of_filters=16;
            $epoch = 50;
        }
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Conv1D(
                $filters=$num_of_filters,#128,
                $kernel_size=3,
                input_shape:[10,1],
                kernel_initializer:'he_normal',
                activation:'relu'),
            $nn->layers()->MaxPooling1D(),
            $nn->layers()->Conv1D($num_of_filters/*128*/,3),
            $nn->layers()->Flatten(),
            $nn->layers()->Activation('softmax'),
        ]);

        $model->compile(
            loss:$nn->losses()->SparseCategoricalCrossEntropy(),
            optimizer:$nn->optimizers()->Adam(),
        );

        // training greater or less
        $x = $mo->array([
            [[0.1],[0.1],[0.2],[0.2],[0.3],[0.3],[0.4],[0.4],[0.5],[0.5]],
            [[0.9],[0.9],[0.8],[0.8],[0.7],[0.7],[0.6],[0.6],[0.5],[0.5]],
            [[0.5],[0.5],[0.6],[0.6],[0.7],[0.7],[0.8],[0.8],[0.9],[0.9]],
            [[0.5],[0.5],[0.4],[0.4],[0.3],[0.3],[0.2],[0.2],[0.1],[0.1]],
        ]);
        $t = $mo->array(
            [1,0,1,0],dtype:NDArray::int32
        );
        $v_x = $mo->array([
            [[0.1],[0.1],[0.25],[0.25],[0.35],[0.35],[0.45],[0.45],[0.6], [0.6] ],
            [[0.9],[0.9],[0.7], [0.7], [0.5], [0.5], [0.3], [0.3], [0.1], [0.1] ],
            [[0.1],[0.1],[0.11],[0.11],[0.12],[0.12],[0.13],[0.13],[0.14],[0.14]],
            [[0.5],[0.5],[0.45],[0.45],[0.4], [0.4], [0.35],[0.35],[0.3], [0.3] ],
        ]);
        $v_t = $mo->array(
            [1,0,1,0],dtype:NDArray::int32
        );
        $history = $model->fit(
            $x,$t,
            epochs:$epoch/*300*/, validation_data:[$v_x,$v_t],verbose:0);

        $this->assertEquals(['loss','accuracy','val_loss','val_accuracy'],array_keys($history));

        if($this->plot) {
            $plt->plot($mo->array($history['loss']),null,null,'loss');
            $plt->plot($mo->array($history['val_loss']),null,null,'val_loss');
            $plt->plot($mo->array($history['accuracy']),null,null,'accuracy');
            $plt->plot($mo->array($history['val_accuracy']),null,null,'val_accuracy');
            $plt->legend();
            $plt->title('ConvolutionMaxPool1D');
            $plt->show();
        }
    }

    public function testFitConv2DandMaxPooling2D()
    {
        $mo = $this->newMatrixOperator();
        if($mo->isAdvanced()) {
            $num_of_filters=128;
            $epoch = 300;
        } else {
            $num_of_filters=8;
            $epoch = 30;
        }
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Conv2D(
                $filters=$num_of_filters,#128,
                $kernel_size=3,
                input_shape:[10,10,1],
                kernel_initializer:'he_normal',
                activation:'relu',
                #activation:'softmax',
                ),
            $nn->layers()->MaxPooling2D(),
            $nn->layers()->Conv2D($num_of_filters/*128*/,3),
            $nn->layers()->Flatten(),
            $nn->layers()->Activation('softmax'),
        ]);

        $model->compile(
            loss:$nn->losses()->SparseCategoricalCrossEntropy(),
            optimizer:$nn->optimizers()->Adam(),
        );

        // training greater or less
        $x = $mo->zeros([4,10,10,1]);
        for($i=0;$i<10;$i++) { $x[0][$i][$i][0]=1.0;}
        for($i=0;$i<10;$i++) { $x[1][$i][9-$i][0]=1.0;}
        for($i=1;$i<9;$i++)  { $x[2][$i+1][$i+1][0]=1.0;}
        for($i=1;$i<9;$i++)  { $x[3][$i+1][9-$i][0]=1.0;}
        $t = $mo->array(
            [1,0,1,0],dtype:NDArray::int32
        );
        $v_x = $mo->zeros([4,10,10,1]);
        for($i=0;$i<8;$i++) { $x[0][$i][$i+2][0]=1.0;}
        for($i=0;$i<8;$i++) { $x[1][$i][9-$i][0]=1.0;}
        for($i=1;$i<8;$i++) { $x[2][$i+1][$i][0]=1.0;}
        for($i=1;$i<8;$i++) { $x[3][$i+2][9-$i][0]=1.0;}
        $v_t = $mo->array(
            [1,0,1,0],dtype:NDArray::int32
        );
        $history = $model->fit(
            $x,$t,
            epochs:$epoch/*100*/,validation_data:[$v_x,$v_t],verbose:0);

        $this->assertEquals(['loss','accuracy','val_loss','val_accuracy'],array_keys($history));

        if($this->plot) {
            $plt->plot($mo->array($history['loss']),null,null,'loss');
            $plt->plot($mo->array($history['val_loss']),null,null,'val_loss');
            $plt->plot($mo->array($history['accuracy']),null,null,'accuracy');
            $plt->plot($mo->array($history['val_accuracy']),null,null,'val_accuracy');
            $plt->legend();
            $plt->title('Convolution2DMaxPool');
            $plt->show();
        }
    }

    public function testFitConv3DandMaxPooling3D()
    {
        $mo = $this->newMatrixOperator();
        if($mo->isAdvanced()) {
            $num_of_filters=128;
            $epoch = 300;
        } else {
            $num_of_filters=8;
            $epoch = 20;
        }
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Conv3D(
                $filters=$num_of_filters,#128,
                $kernel_size=3,
                input_shape:[10,10,10,1],
                kernel_initializer:'he_normal',
                activation:'relu'),
            $nn->layers()->MaxPooling3D(),
            $nn->layers()->Conv3D($num_of_filters/*128*/,3),
            $nn->layers()->Flatten(),
            $nn->layers()->Activation('softmax'),
        ]);

        $model->compile(
            loss:$nn->losses()->SparseCategoricalCrossEntropy(),
        );

        // training greater or less
        $x = $mo->zeros([4,10,10,10,1]);
        for($i=0;$i<10;$i++) { $x[0][$i][$i][$i][0]=1.0;}
        for($i=0;$i<10;$i++) { $x[1][$i][$i][9-$i][0]=1.0;}
        for($i=1;$i<9;$i++)  { $x[2][$i][$i+1][$i+1][0]=1.0;}
        for($i=1;$i<9;$i++)  { $x[3][$i][$i+1][9-$i][0]=1.0;}
        $t = $mo->array(
            [1,0,1,0],dtype:NDArray::int32
        );
        $v_x = $mo->zeros([4,10,10,10,1]);
        for($i=0;$i<8;$i++) { $x[0][$i][$i][$i+2][0]=1.0;}
        for($i=0;$i<8;$i++) { $x[1][$i][$i][9-$i][0]=1.0;}
        for($i=1;$i<8;$i++) { $x[2][$i][$i+1][$i][0]=1.0;}
        for($i=1;$i<8;$i++) { $x[3][$i][$i+2][9-$i][0]=1.0;}
        $v_t = $mo->array(
            [1,0,1,0],dtype:NDArray::int32
        );
        $history = $model->fit($x,$t,epochs:$epoch/*100*/,validation_data:[$v_x,$v_t],verbose:0);

        $this->assertEquals(['loss','accuracy','val_loss','val_accuracy'],array_keys($history));

        if($this->plot) {
            $plt->plot($mo->array($history['loss']),null,null,'loss');
            $plt->plot($mo->array($history['val_loss']),null,null,'val_loss');
            $plt->plot($mo->array($history['accuracy']),null,null,'accuracy');
            $plt->plot($mo->array($history['val_accuracy']),null,null,'val_accuracy');
            $plt->legend();
            $plt->title('Convolution3DMaxPool');
            $plt->show();
        }
    }

    public function testFitConv1DandAveragePooling1D()
    {
        $mo = $this->newMatrixOperator();
        if($mo->isAdvanced()) {
            $num_of_filters=128;
            $epoch = 300;
        } else {
            $num_of_filters=16;
            $epoch = 50;
        }
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Conv1D(
                $filters=$num_of_filters,#128,
                $kernel_size=3,
                input_shape:[10,1],
                kernel_initializer:'he_normal',
                activation:'relu'),
            $nn->layers()->AveragePooling1D(),
            $nn->layers()->Conv1D($num_of_filters/*128*/,3),
            $nn->layers()->Flatten(),
            $nn->layers()->Activation('softmax'),
        ]);

        $model->compile(
            loss:$nn->losses()->SparseCategoricalCrossEntropy(),
            optimizer:$nn->optimizers()->Adam(),
        );

        // training greater or less
        $x = $mo->array([
            [[0.1],[0.1],[0.2],[0.2],[0.3],[0.3],[0.4],[0.4],[0.5],[0.5]],
            [[0.9],[0.9],[0.8],[0.8],[0.7],[0.7],[0.6],[0.6],[0.5],[0.5]],
            [[0.5],[0.5],[0.6],[0.6],[0.7],[0.7],[0.8],[0.8],[0.9],[0.9]],
            [[0.5],[0.5],[0.4],[0.4],[0.3],[0.3],[0.2],[0.2],[0.1],[0.1]],
        ]);
        $t = $mo->array(
            [1,0,1,0],dtype:NDArray::int32
        );
        $v_x = $mo->array([
            [[0.1],[0.1],[0.25],[0.25],[0.35],[0.35],[0.45],[0.45],[0.6], [0.6] ],
            [[0.9],[0.9],[0.7], [0.7], [0.5], [0.5], [0.3], [0.3], [0.1], [0.1] ],
            [[0.1],[0.1],[0.11],[0.11],[0.12],[0.12],[0.13],[0.13],[0.14],[0.14]],
            [[0.5],[0.5],[0.45],[0.45],[0.4], [0.4], [0.35],[0.35],[0.3], [0.3] ],
        ]);
        $v_t = $mo->array(
            [1,0,1,0],dtype:NDArray::int32
        );
        $history = $model->fit(
            $x,$t,
            epochs:$epoch/*300*/,validation_data:[$v_x,$v_t],verbose:0);

        $this->assertEquals(['loss','accuracy','val_loss','val_accuracy'],array_keys($history));

        if($this->plot) {
            $plt->plot($mo->array($history['loss']),null,null,'loss');
            $plt->plot($mo->array($history['val_loss']),null,null,'val_loss');
            $plt->plot($mo->array($history['accuracy']),null,null,'accuracy');
            $plt->plot($mo->array($history['val_accuracy']),null,null,'val_accuracy');
            $plt->legend();
            $plt->title('ConvolutionAvgPool1D');
            $plt->show();
        }
    }

    public function testFitConv2DandAveragePooling2D()
    {
        $mo = $this->newMatrixOperator();
        if($mo->isAdvanced()) {
            $num_of_filters=128;
            $epoch = 300;
        } else {
            $num_of_filters=8;
            $epoch = 30;
        }
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Conv2D(
                $filters=$num_of_filters,#128,
                $kernel_size=3,
                input_shape:[10,10,1],
                kernel_initializer:'he_normal',
                activation:'relu',
                #activation:'softmax',
                ),
            $nn->layers()->AveragePooling2D(),
            $nn->layers()->Conv2D($num_of_filters/*128*/,3),
            $nn->layers()->Flatten(),
            $nn->layers()->Activation('softmax'),
        ]);

        $model->compile(
            loss:$nn->losses()->SparseCategoricalCrossEntropy(),
            optimizer:$nn->optimizers()->Adam(),
        );

        // training greater or less
        $x = $mo->zeros([4,10,10,1]);
        for($i=0;$i<10;$i++) { $x[0][$i][$i][0]=1.0;}
        for($i=0;$i<10;$i++) { $x[1][$i][9-$i][0]=1.0;}
        for($i=1;$i<9;$i++)  { $x[2][$i+1][$i+1][0]=1.0;}
        for($i=1;$i<9;$i++)  { $x[3][$i+1][9-$i][0]=1.0;}
        $t = $mo->array(
            [1,0,1,0],dtype:NDArray::int32
        );
        $v_x = $mo->zeros([4,10,10,1]);
        for($i=0;$i<8;$i++) { $x[0][$i][$i+2][0]=1.0;}
        for($i=0;$i<8;$i++) { $x[1][$i][9-$i][0]=1.0;}
        for($i=1;$i<8;$i++) { $x[2][$i+1][$i][0]=1.0;}
        for($i=1;$i<8;$i++) { $x[3][$i+2][9-$i][0]=1.0;}
        $v_t = $mo->array(
            [1,0,1,0],dtype:NDArray::int32
        );
        $history = $model->fit(
            $x,$t,
            epochs:$epoch,validation_data:[$v_x,$v_t],verbose:0);

        $this->assertEquals(['loss','accuracy','val_loss','val_accuracy'],array_keys($history));

        if($this->plot) {
            $plt->plot($mo->array($history['loss']),null,null,'loss');
            $plt->plot($mo->array($history['val_loss']),null,null,'val_loss');
            $plt->plot($mo->array($history['accuracy']),null,null,'accuracy');
            $plt->plot($mo->array($history['val_accuracy']),null,null,'val_accuracy');
            $plt->legend();
            $plt->title('Convolution2DAvgPool');
            $plt->show();
        }
    }

    public function testFitConv3DandAveragePooling3D()
    {
        $mo = $this->newMatrixOperator();
        if($mo->isAdvanced()) {
            $num_of_filters=128;
            $epoch = 300;
        } else {
            $num_of_filters=8;
            $epoch = 20;
        }
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Conv3D(
                $filters=$num_of_filters,#128,
                $kernel_size=3,
                input_shape:[10,10,10,1],
                kernel_initializer:'he_normal',
                activation:'relu'),
            $nn->layers()->AveragePooling3D(),
            $nn->layers()->Conv3D($num_of_filters/*128*/,3),
            $nn->layers()->Flatten(),
            $nn->layers()->Activation('softmax'),
        ]);

        $model->compile(
            loss:$nn->losses()->SparseCategoricalCrossEntropy(),
        );

        // training greater or less
        $x = $mo->zeros([4,10,10,10,1]);
        for($i=0;$i<10;$i++) { $x[0][$i][$i][$i][0]=1.0;}
        for($i=0;$i<10;$i++) { $x[1][$i][$i][9-$i][0]=1.0;}
        for($i=1;$i<9;$i++)  { $x[2][$i][$i+1][$i+1][0]=1.0;}
        for($i=1;$i<9;$i++)  { $x[3][$i][$i+1][9-$i][0]=1.0;}
        $t = $mo->array(
            [1,0,1,0],dtype:NDArray::int32
        );
        $v_x = $mo->zeros([4,10,10,10,1]);
        for($i=0;$i<8;$i++) { $x[0][$i][$i][$i+2][0]=1.0;}
        for($i=0;$i<8;$i++) { $x[1][$i][$i][9-$i][0]=1.0;}
        for($i=1;$i<8;$i++) { $x[2][$i][$i+1][$i][0]=1.0;}
        for($i=1;$i<8;$i++) { $x[3][$i][$i+2][9-$i][0]=1.0;}
        $v_t = $mo->array(
            [1,0,1,0],dtype:NDArray::int32
        );
        $history = $model->fit($x,$t,epochs:$epoch/*100*/,validation_data:[$v_x,$v_t],verbose:0);

        $this->assertEquals(['loss','accuracy','val_loss','val_accuracy'],array_keys($history));

        if($this->plot) {
            $plt->plot($mo->array($history['loss']),null,null,'loss');
            $plt->plot($mo->array($history['val_loss']),null,null,'val_loss');
            $plt->plot($mo->array($history['accuracy']),null,null,'accuracy');
            $plt->plot($mo->array($history['val_accuracy']),null,null,'val_accuracy');
            $plt->legend();
            $plt->title('Convolution3DAvgPool');
            $plt->show();
        }
    }

    public function testFitEmbedding()
    {
        $mo = $this->newMatrixOperator();
        if($mo->isAdvanced()) {
            $epoch = 300;
        } else {
            $epoch = 50;
        }
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Embedding($inputDim=10,$outputDim=10,
                    input_length:4,
                    #'kernel_initializer'=>'glorot_normal',
            ),
            $nn->layers()->Activation('softmax'),
        ]);

        $model->compile(
            loss:$nn->losses()->SparseCategoricalCrossEntropy(),
            optimizer:'adam',
        );
        //$model->summary();

        // training sequences
        $x = $mo->array([
            [0,1,2,9],
            [9,8,7,6],
            [1,3,3,4],
            [5,4,3,2],
        ]);
        $t = $mo->array([
            [0,1,2,9],
            [9,8,7,6],
            [1,3,3,4],
            [5,4,3,2],
        ],dtype:NDArray::int32);
        $v_x = $mo->array([
            [2,3,3,4],
            [1,1,1,4],
            [4,3,3,1],
            [9,3,3,2],
        ]);
        $v_t = $mo->array([
            [2,3,3,4],
            [1,1,1,4],
            [4,3,3,1],
            [9,3,3,2],
        ],dtype:NDArray::int32);

        $history = $model->fit($x,$t,epochs:$epoch/*300*/,batch_size:1,validation_data:[$v_x,$v_t],verbose:0);

        $this->assertEquals(['loss','accuracy','val_loss','val_accuracy'],array_keys($history));

        if($this->plot) {
            $plt->plot($mo->array($history['loss']),null,null,'loss');
            $plt->plot($mo->array($history['val_loss']),null,null,'val_loss');
            $plt->plot($mo->array($history['accuracy']),null,null,'accuracy');
            $plt->plot($mo->array($history['val_accuracy']),null,null,'val_accuracy');
            $plt->legend();
            $plt->title('Embedding');
            $plt->show();
        }
    }

    public function testFitSimpleRNN()
    {
        $mo = $this->newMatrixOperator();
        if($mo->isAdvanced()) {
            $epoch = 300;
        } else {
            $epoch = 100;
        }
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->SimpleRNN(
                $units=16,
                    input_shape:[4,10],
                    #input_shape:[1,10],
                    #kernel_initializer:'glorot_normal',
                    #recurrent_initializer:'glorot_normal',
                    #recurrent_initializer:'glorot_normal',
            ),
            $nn->layers()->Activation('softmax'),
        ]);

        $model->compile(
            loss:$nn->losses()->SparseCategoricalCrossEntropy(),
            optimizer:'adam',
        );
        #$model->summary();

        // training up and down
        $x = $mo->array([
            [0,1,2,9],
            [9,8,7,6],
            [1,3,3,4],
            [5,4,3,2],
        ],dtype:NDArray::int32);
        $t = $mo->array(
            [1,0,1,0],dtype:NDArray::int32
        );
        $v_x = $mo->array([
            [2,3,3,4],
            [1,1,1,4],
            [4,3,3,1],
            [9,3,3,2],
        ],dtype:NDArray::int32);
        $v_t = $mo->array(
            [1,1,0,0],dtype:NDArray::int32
        );
        $x = $mo->la()->onehot($x->reshape([16]),$numClass=10)->reshape([4,4,10]);
        $v_x = $mo->la()->onehot($v_x->reshape([16]),$numClass=10)->reshape([4,4,10]);

        $history = $model->fit($x,$t,epochs:$epoch/*300*/,batch_size:1,validation_data:[$v_x,$v_t],verbose:0);

        $this->assertEquals(['loss','accuracy','val_loss','val_accuracy'],array_keys($history));

        if($this->plot) {
            $plt->plot($mo->array($history['loss']),null,null,'loss');
            $plt->plot($mo->array($history['val_loss']),null,null,'val_loss');
            $plt->plot($mo->array($history['accuracy']),null,null,'accuracy');
            $plt->plot($mo->array($history['val_accuracy']),null,null,'val_accuracy');
            $plt->legend();
            $plt->title('SimpleRNN');
            $plt->show();
        }
    }

    public function testFitSimpleRNNRetSeq()
    {
        $mo = $this->newMatrixOperator();
        if($mo->isAdvanced()) {
            $epoch = 300;
        } else {
            $epoch = 100;
        }
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->SimpleRNN(
                $units=10,
                input_shape:[4,10],
                #input_shape:[1,10],
                #kernel_initializer:'glorot_normal',
                #recurrent_initializer:'glorot_normal',
                #recurrent_initializer:'glorot_normal',
                return_sequences:true,
            ),
            $nn->layers()->Activation('softmax'),
        ]);

        $model->compile(
            loss:$nn->losses()->SparseCategoricalCrossEntropy(),
            optimizer:'adam',
        );
        #$model->summary();

        // training sequences
        $x = $mo->array([
            [0,1,2,9],
            [9,8,7,6],
            [1,3,3,4],
            [5,4,3,2],
        ],dtype:NDArray::int32);
        $t = $mo->array([
            [0,1,2,9],
            [9,8,7,6],
            [1,3,3,4],
            [5,4,3,2],
        ],dtype:NDArray::int32);
        $v_x = $mo->array([
            [2,3,3,4],
            [1,1,1,4],
            [4,3,3,1],
            [9,3,3,2],
        ],dtype:NDArray::int32);
        $v_t = $mo->array([
            [2,3,3,4],
            [1,1,1,4],
            [4,3,3,1],
            [9,3,3,2],
        ],dtype:NDArray::int32);
        $x = $mo->la()->onehot($x->reshape([16]),$numClass=10)->reshape([4,4,10]);
        $v_x = $mo->la()->onehot($v_x->reshape([16]),$numClass=10)->reshape([4,4,10]);

        $callback = new WeightLog($mo);
        $history = $model->fit($x,$t,epochs:$epoch/*300*/,batch_size:1,validation_data:[$v_x,$v_t],verbose:0,callbacks:[$callback]);

        $this->assertEquals(['loss','accuracy','val_loss','val_accuracy'],array_keys($history));

        if($this->plot) {
            $plt->plot($mo->array($history['loss']),null,null,'loss');
            $plt->plot($mo->array($history['val_loss']),null,null,'val_loss');
            $plt->plot($mo->array($history['accuracy']),null,null,'accuracy');
            $plt->plot($mo->array($history['val_accuracy']),null,null,'val_accuracy');
            $plt->legend();
            $plt->title('SimpleRNN return_sequences');
            $plt->figure();
            foreach ($callback->getGradlog() as $key => $log) {
                $plt->plot($mo->array($log),null,null,$label=$key);
            }
            $plt->legend();
            $plt->show();
        }
    }

    public function testFitSimSimpleRNN()
    {
        $mo = $this->newMatrixOperator();
        if($mo->isAdvanced()) {
            $epoch = 300;
        } else {
            $epoch = 100;
        }
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense(
                $units=16,
                #input_shape:[4,10],
                input_shape:[10],
                #kernel_initializer:'glorot_normal',
                #recurrent_initializer:'glorot_normal',
                #recurrent_initializer:'glorot_normal',
                #activation:'sigmoid',
                activation:'tanh',
            ),
            $nn->layers()->Activation('softmax'),
        ]);

        $model->compile(
            loss:'sparse_categorical_crossentropy',
            optimizer:'adam',
            #optimizer'sgd',
        );
        #$model->summary();

        $x = $mo->array([
            [0],
            [9],
            [1],
            [5],
        ],dtype:NDArray::int32);
        $t = $mo->array(
            [0,9,1,5],dtype:NDArray::int32
        );
        $v_x = $mo->array([
            [2],
            [1],
            [4],
            [9],
        ],dtype:NDArray::int32);
        $v_t = $mo->array(
            [2,1,4,9],dtype:NDArray::int32
        );
        $x = $mo->la()->onehot($x->reshape([4]),$numClass=10)->reshape([4,10]);
        $v_x = $mo->la()->onehot($v_x->reshape([4]),$numClass=10)->reshape([4,10]);

        $history = $model->fit($x,$t,epochs:$epoch/*300*/,batch_size:1,validation_data:[$v_x,$v_t],verbose:0);

        $this->assertEquals(['loss','accuracy','val_loss','val_accuracy'],array_keys($history));

        if($this->plot) {
            $plt->plot($mo->array($history['loss']),null,null,'loss');
            $plt->plot($mo->array($history['val_loss']),null,null,'val_loss');
            $plt->plot($mo->array($history['accuracy']),null,null,'accuracy');
            $plt->plot($mo->array($history['val_accuracy']),null,null,'val_accuracy');
            $plt->legend();
            $plt->title('SimSingleRNN');
            $plt->show();
        }
    }
    public function testFitLSTM()
    {
        $mo = $this->newMatrixOperator();
        if($mo->isAdvanced()) {
            $epoch = 300;
        } else {
            $epoch = 100;
        }
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->LSTM(
                $units=16,
                    input_shape:[4,10],
                ),
                $nn->layers()->Activation('softmax'),
        ]);

        $model->compile(
            loss:$nn->losses()->SparseCategoricalCrossEntropy(),
            optimizer:$nn->optimizers()->Adam(),
        );
        //$model->summary();

        // training up and down
        $x = $mo->array([
            [0,1,2,9],
            [9,8,7,6],
            [1,3,3,4],
            [5,4,3,2],
        ],dtype:NDArray::int32);
        $t = $mo->array(
            [1,0,1,0],dtype:NDArray::int32
        );
        $v_x = $mo->array([
            [2,3,3,4],
            [1,1,1,4],
            [4,3,3,1],
            [9,3,3,2],
        ],dtype:NDArray::int32);
        $v_t = $mo->array(
            [1,1,0,0],dtype:NDArray::int32
        );
        $x = $mo->la()->onehot($x->reshape([16]),$numClass=10)->reshape([4,4,10]);
        $v_x = $mo->la()->onehot($v_x->reshape([16]),$numClass=10)->reshape([4,4,10]);
        $history = $model->fit($x,$t,epochs:$epoch/*300*/,batch_size:1,validation_data:[$v_x,$v_t],verbose:0);

        $this->assertEquals(['loss','accuracy','val_loss','val_accuracy'],array_keys($history));

        if($this->plot) {
            $plt->plot($mo->array($history['loss']),null,null,'loss');
            $plt->plot($mo->array($history['val_loss']),null,null,'val_loss');
            $plt->plot($mo->array($history['accuracy']),null,null,'accuracy');
            $plt->plot($mo->array($history['val_accuracy']),null,null,'val_accuracy');
            $plt->legend();
            $plt->title('LSTM');
            $plt->show();
        }
    }

    public function testFitLSTMRetSeq()
    {
        $mo = $this->newMatrixOperator();
        if($mo->isAdvanced()) {
            $epoch = 300;
        } else {
            $epoch = 100;
        }
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->LSTM(
                $units=16,
                    input_shape:[4,10],
                    return_sequences:true,
                ),
                $nn->layers()->Activation('softmax'),
        ]);

        $model->compile(
            loss:$nn->losses()->SparseCategoricalCrossEntropy(),
            optimizer:$nn->optimizers()->Adam(),
        );
        //$model->summary();

        // training up and down
        $x = $mo->array([
            [0,1,2,9],
            [9,8,7,6],
            [1,3,3,4],
            [5,4,3,2],
        ],dtype:NDArray::int32);
        $t = $mo->array([
            [0,1,2,9],
            [9,8,7,6],
            [1,3,3,4],
            [5,4,3,2],
        ],dtype:NDArray::int32);
        $v_x = $mo->array([
            [2,3,3,4],
            [1,1,1,4],
            [4,3,3,1],
            [9,3,3,2],
        ],dtype:NDArray::int32);
        $v_t = $mo->array([
            [2,3,3,4],
            [1,1,1,4],
            [4,3,3,1],
            [9,3,3,2],
        ],dtype:NDArray::int32);
        $x = $mo->la()->onehot($x->reshape([16]),$numClass=10)->reshape([4,4,10]);
        $v_x = $mo->la()->onehot($v_x->reshape([16]),$numClass=10)->reshape([4,4,10]);
        $history = $model->fit($x,$t,epochs:$epoch/*300*/,batch_size:1,validation_data:[$v_x,$v_t],verbose:0);

        $this->assertEquals(['loss','accuracy','val_loss','val_accuracy'],array_keys($history));

        if($this->plot) {
            $plt->plot($mo->array($history['loss']),null,null,'loss');
            $plt->plot($mo->array($history['val_loss']),null,null,'val_loss');
            $plt->plot($mo->array($history['accuracy']),null,null,'accuracy');
            $plt->plot($mo->array($history['val_accuracy']),null,null,'val_accuracy');
            $plt->legend();
            $plt->title('LSTM return_sequences');
            $plt->show();
        }
    }

    public function testFitGRUDefault()
    {
        $mo = $this->newMatrixOperator();
        if($mo->isAdvanced()) {
            $epoch = 300;
        } else {
            $epoch = 100;
        }
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->GRU(
                $units=16,
                input_shape:[4,10],
                ),
                $nn->layers()->Activation('softmax'),
        ]);

        $model->compile(
            loss:$nn->losses()->SparseCategoricalCrossEntropy(),
            optimizer:'adam',
        );
        #$model->summary();

        // training up and down
        $x = $mo->array([
            [0,1,2,9],
            [9,8,7,6],
            [1,3,3,4],
            [5,4,3,2],
        ],dtype:NDArray::int32);
        $t = $mo->array(
            [1,0,1,0],dtype:NDArray::int32
        );
        $v_x = $mo->array([
            [2,3,3,4],
            [1,1,1,4],
            [4,3,3,1],
            [9,3,3,2],
        ],dtype:NDArray::int32);
        $v_t = $mo->array(
            [1,1,0,0],dtype:NDArray::int32
        );
        $x = $mo->la()->onehot($x->reshape([16]),$numClass=10)->reshape([4,4,10]);
        $v_x = $mo->la()->onehot($v_x->reshape([16]),$numClass=10)->reshape([4,4,10]);
        $history = $model->fit($x,$t,epochs:$epoch/*300*/,batch_size:1,validation_data:[$v_x,$v_t],verbose:0);

        $this->assertEquals(['loss','accuracy','val_loss','val_accuracy'],array_keys($history));

        if($this->plot) {
            $plt->plot($mo->array($history['loss']),null,null,'loss');
            $plt->plot($mo->array($history['val_loss']),null,null,'val_loss');
            $plt->plot($mo->array($history['accuracy']),null,null,'accuracy');
            $plt->plot($mo->array($history['val_accuracy']),null,null,'val_accuracy');
            $plt->legend();
            $plt->title('GRU');
            $plt->show();
        }
    }

    public function testFitGRUWithoutResetAfter()
    {
        $mo = $this->newMatrixOperator();
        if($mo->isAdvanced()) {
            $epoch = 300;
        } else {
            $epoch = 100;
        }
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->GRU(
                $units=16,
                input_shape:[4,10],
                reset_after:false,
                ),
                $nn->layers()->Activation('softmax'),
        ]);

        $model->compile(
            loss:$nn->losses()->SparseCategoricalCrossEntropy(),
            optimizer:'adam',
        );
        #$model->summary();

        // training up and down
        $x = $mo->array([
            [0,1,2,9],
            [9,8,7,6],
            [1,3,3,4],
            [5,4,3,2],
        ],dtype:NDArray::int32);
        $t = $mo->array(
            [1,0,1,0],dtype:NDArray::int32
        );
        $v_x = $mo->array([
            [2,3,3,4],
            [1,1,1,4],
            [4,3,3,1],
            [9,3,3,2],
        ],dtype:NDArray::int32);
        $v_t = $mo->array(
            [1,1,0,0],dtype:NDArray::int32
        );
        $x = $mo->la()->onehot($x->reshape([16]),$numClass=10)->reshape([4,4,10]);
        $v_x = $mo->la()->onehot($v_x->reshape([16]),$numClass=10)->reshape([4,4,10]);
        $history = $model->fit($x,$t,epochs:$epoch/*300*/,batch_size:1,validation_data:[$v_x,$v_t],verbose:0);

        $this->assertEquals(['loss','accuracy','val_loss','val_accuracy'],array_keys($history));

        if($this->plot) {
            $plt->plot($mo->array($history['loss']),null,null,'loss');
            $plt->plot($mo->array($history['val_loss']),null,null,'val_loss');
            $plt->plot($mo->array($history['accuracy']),null,null,'accuracy');
            $plt->plot($mo->array($history['val_accuracy']),null,null,'val_accuracy');
            $plt->legend();
            $plt->title('GRU without reset_after');
            $plt->show();
        }
    }

    public function testFitGRURetSeq()
    {
        $mo = $this->newMatrixOperator();
        if($mo->isAdvanced()) {
            $epoch = 300;
        } else {
            $epoch = 100;
        }
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->GRU(
                $units=10,
                input_shape:[4,10],
                #input_shape:[1,10],
                #kernel_initializer:'glorot_normal',
                #recurrent_initializer:'glorot_normal',
                #recurrent_initializer:'glorot_normal',
                return_sequences:true,
            ),
            $nn->layers()->Activation('softmax'),
        ]);

        $model->compile(
            loss:$nn->losses()->SparseCategoricalCrossEntropy(),
            optimizer:'adam',
        );
        #$model->summary();

        // training sequences
        $x = $mo->array([
            [0,1,2,9],
            [9,8,7,6],
            [1,3,3,4],
            [5,4,3,2],
        ],dtype:NDArray::int32);
        $t = $mo->array([
            [0,1,2,9],
            [9,8,7,6],
            [1,3,3,4],
            [5,4,3,2],
        ],dtype:NDArray::int32);
        $v_x = $mo->array([
            [2,3,3,4],
            [1,1,1,4],
            [4,3,3,1],
            [9,3,3,2],
        ],dtype:NDArray::int32);
        $v_t = $mo->array([
            [2,3,3,4],
            [1,1,1,4],
            [4,3,3,1],
            [9,3,3,2],
        ],dtype:NDArray::int32);
        $x = $mo->la()->onehot($x->reshape([16]),$numClass=10)->reshape([4,4,10]);
        $v_x = $mo->la()->onehot($v_x->reshape([16]),$numClass=10)->reshape([4,4,10]);

        $callback = new WeightLog($mo);
        $history = $model->fit($x,$t,epochs:$epoch/*300*/,batch_size:1,validation_data:[$v_x,$v_t],verbose:0,callbacks:[$callback]);

        $this->assertEquals(['loss','accuracy','val_loss','val_accuracy'],array_keys($history));

        if($this->plot) {
            $plt->plot($mo->array($history['loss']),null,null,'loss');
            $plt->plot($mo->array($history['val_loss']),null,null,'val_loss');
            $plt->plot($mo->array($history['accuracy']),null,null,'accuracy');
            $plt->plot($mo->array($history['val_accuracy']),null,null,'val_accuracy');
            $plt->legend();
            $plt->title('GRU return_sequences');
            $plt->figure();
            foreach ($callback->getGradlog() as $key => $log) {
                $plt->plot($mo->array($log),null,null,$label=$key);
            }
            $plt->legend();
            $plt->show();
        }
    }

    public function testFitRepeatVector()
    {
        $mo = $this->newMatrixOperator();
        if($mo->isAdvanced()) {
            $epoch = 300;
        } else {
            $epoch = 50;
        }
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense(
                $units=16,
                input_shape:[4,10],
                ),
                $nn->layers()->Flatten(),
                $nn->layers()->RepeatVector(3),
                $nn->layers()->Dense(10),
                $nn->layers()->Activation('softmax'),
        ]);

        $model->compile(
            loss:$nn->losses()->SparseCategoricalCrossEntropy(),
            optimizer:'adam',
        );
        #$model->summary();

        // training up and down
        $x = $mo->array([
            [0,1,2,9],
            [9,8,7,6],
            [1,3,3,4],
            [5,4,3,2],
        ],dtype:NDArray::int32);
        $t = $mo->array([
            [0,1,0],
            [1,0,1],
            [0,1,0],
            [1,0,1],
        ],dtype:NDArray::int32);
        $v_x = $mo->array([
            [2,3,3,4],
            [1,1,1,4],
            [4,3,3,1],
            [9,3,3,2],
        ],dtype:NDArray::int32);
        $v_t = $mo->array([
            [0,1,0],
            [0,1,0],
            [1,0,1],
            [1,0,1],
        ],dtype:NDArray::int32);
        $x = $mo->la()->onehot($x->reshape([16]),$numClass=10)->reshape([4,4,10]);
        $v_x = $mo->la()->onehot($v_x->reshape([16]),$numClass=10)->reshape([4,4,10]);
        $history = $model->fit($x,$t,epochs:$epoch/*300*/,batch_size:1,validation_data:[$v_x,$v_t],verbose:0);

        $this->assertEquals(['loss','accuracy','val_loss','val_accuracy'],array_keys($history));

        if($this->plot) {
            $plt->plot($mo->array($history['loss']),null,null,'loss');
            $plt->plot($mo->array($history['val_loss']),null,null,'val_loss');
            $plt->plot($mo->array($history['accuracy']),null,null,'accuracy');
            $plt->plot($mo->array($history['val_accuracy']),null,null,'val_accuracy');
            $plt->legend();
            $plt->title('RepeatVector');
            $plt->show();
        }
    }

    public function testToJson()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,input_shape:[2]),
            $nn->layers()->BatchNormalization(),
            $nn->layers()->Activation('sigmoid'),
            $nn->layers()->Dense($units=2,activation:'softmax'),
        ]);
        $model->compile();
        $json = $model->toJson();
        $this->assertTrue(true);
    }

    public function testSaveAndLoadWeightsNormal()
    {
        $mo = $this->newMatrixOperator();
        if($mo->isAdvanced()) {
            $epoch = 300;
        } else {
            $epoch = 50;
        }
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,input_shape:[2]),
            $nn->layers()->BatchNormalization(),
            $nn->layers()->Activation('sigmoid'),
            $nn->layers()->Dense($units=2,activation:'softmax'),
        ]);
        $model->compile();
        $x = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $t = $mo->array([0, 0, 0, 1, 1, 1],dtype:NDArray::int32);
        $history = $model->fit($x,$t,epochs:$epoch,verbose:0);
        $logs = $model->evaluate($x,$t);

        $origY = $model->predict($x);
        $this->assertCount(8,$model->variables());
        $this->assertCount(6,$model->trainableVariables());
        $origVariables = $model->variables();
        $origLayers = $model->layers();

        // save config and weights
        $json = $model->toJson();
        $weights = [];
        $model->saveWeights($weights);
        $config = json_decode($json,true);

        // ****************************************************
        // new model from config and load weights
        $model = $nn->Models->modelFromConfig($config);
        $model->loadWeights($weights);

        $logs2 = $model->evaluate($x,$t);
        $this->assertLessThan(0.5,abs($logs['loss']-$logs2['loss']));
        $this->assertLessThan(0.5,abs($logs['accuracy']-$logs2['accuracy']));

        $y = $model->predict($x);
        $this->assertCount(8,$model->variables());
        $this->assertCount(6,$model->trainableVariables());
        $variables = $model->variables();

        $la = $K->localLA();
        foreach(array_map(null,$variables,$origVariables) as [$v,$origV]) {
            $v = $K->ndarray($v);
            $origV = $K->ndarray($origV);
            $this->assertTrue($la->isclose($v,$origV));
        }

        //// orig object check
        $params = [];
        $idx = 0;
        foreach($origLayers as $ly) {
            foreach($ly->getParams() as $v) {
                $idx++;
                $params[] = $v;
            }
        }
        foreach(array_map(null,$origVariables,$params) as $idx => [$v,$p]) {
            $this->assertEquals(spl_object_id($v->value()),spl_object_id($p));
        }

        //// loaded object check
        $layers = $model->layers();
        $params = [];
        $idx = 0;
        foreach($layers as $ly) {
            foreach($ly->getParams() as $v) {
                $idx++;
                $params[] = $v;
            }
        }
        foreach(array_map(null,$variables,$params) as $idx => [$v,$p]) {
            $this->assertEquals(spl_object_id($v->value()),spl_object_id($p));
        }

        foreach(array_map(null,$layers,$origLayers) as [$ly,$origLy]) {
            foreach(array_map(null,$ly->getParams(),$origLy->getParams()) as [$v,$origV]) {
                $v = $K->ndarray($v);
                $origV = $K->ndarray($origV);
                $this->assertTrue($la->isclose($v,$origV));
            }
        }
        $diff = $la->max($la->square($la->axpy($origY,$la->copy($y),-1)));
        $this->assertLessThan(1.0e-6,$diff);
    }

    public function testClone()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();

        $origModel = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,input_shape:[2]),
            $nn->layers()->BatchNormalization(),
            $nn->layers()->Activation('sigmoid'),
            $nn->layers()->Dense($units=2,activation:'softmax'),
        ]);

        $origModel->compile();
        $inputs = $mo->zeros([1,2]);
        $trues = $mo->array([0],NDArray::int32);
        $origModel->fit($inputs,$trues,epochs:1,batch_size:1,verbose:0);
        $model = clone $origModel;

        $this->assertCount(2+4+2,$origModel->variables());
        $this->assertCount(2+2+2,$origModel->trainableVariables());
        $this->assertCount(2+4+2,$model->variables());
        $this->assertCount(2+2+2,$model->trainableVariables());

        $origVars = $origModel->trainableVariables();
        $vars = $model->trainableVariables();
        $this->assertCount(2+2+2,$vars);
        foreach (array_map(null,$origVars,$vars) as [$origvar,$destvar]) {
            $this->assertNotEquals(spl_object_id($origvar),spl_object_id($destvar));
            $this->assertNotEquals(spl_object_id($origvar->value()),spl_object_id($destvar->value()));
        }
    }

    public function testUseInCustomModel()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();

        $seq = $nn->models->Sequential();
        $seq->add($nn->layers->Dense(2,
            input_shape:[3],activation:'softmax'));
        $model = new TestCustomModel($nn,$seq);
        $model->compile();
        //$model->summary();
        $parms = $model->trainableVariables();
        $this->assertCount(2,$parms);
        $model->fit($mo->zeros([5,3]),$mo->zeros([5],dtype:NDArray::int32),
            epochs:1, verbose:0);
        $this->assertEquals([3,2],$parms[0]->shape());
        $this->assertEquals([2],$parms[1]->shape());
        $predicts = $model->predict($mo->zeros([5,3]));
        $this->assertEquals([5,2],$predicts->shape());
        $res = $model->evaluate($mo->zeros([5,3]),$mo->zeros([5],dtype:NDArray::int32));
        $this->assertTrue(true);
    }

    //
    //  The function of organizing models has been moved to 
    //  Rindow\NeuralNetworks\Gradient\Core\Modules
    //  
    //
    //public function testAddCustomModel()
    //{
    //    Dense::$nameNumbering = 0;
    //    $mo = $this->newMatrixOperator();
    //    $nn = $this->newNeuralNetworks($mo);
    //    $K = $nn->backend();
    //    $g = $nn->gradient();
    //
    //    $seq = $nn->models->Sequential();
    //    $seq->add(new TestCustomSubModel($nn));
    //    $seq->add(new TestCustomSubModel($nn));
    //    // Raw data
    //    $x = $K->array([[2],[3]]);
    //    $y = $seq($x);
    //    $this->assertEquals("[[16],[81]]",$mo->toString($y));
    //    // Pure model
    //    $x = $g->Variable($x);
    //    $y = $seq($x);
    //    $this->assertEquals("[[16],[81]]",$mo->toString($y));
    //    // Gradient on pure model
    //    $y = $nn->with($tape=$g->GradientTape(),fn() =>
    //        $seq($x)
    //    );
    //    $this->assertEquals("[[16],[81]]",$mo->toString($y));
    //    $this->assertEquals("[[32],[108]]",$mo->toString($tape->gradient($y,$x)));
    //    // In function graph
    //    $func = $g->Function(fn($x) =>
    //        $seq($x)
    //    );
    //    // ...  building graph
    //    $y = $func($x);
    //    $this->assertEquals("[[16],[81]]",$mo->toString($y));
    //    // ...  execute graph
    //    $y = $func($x);
    //    $this->assertEquals("[[16],[81]]",$mo->toString($y));
    //    // Gradient on graph
    //    $y = $nn->with($tape=$g->GradientTape(),fn() =>
    //        $func($x)
    //    );
    //    $this->assertEquals("[[16],[81]]",$mo->toString($y));
    //    $this->assertEquals("[[32],[108]]",$mo->toString($tape->gradient($y,$x)));
    //
    //    // get params
    //    $params =  $seq->trainableVariables();
    //    $this->assertCount(6,$params);
    //
    //    $layerParams = $nn->layers->Dense(10)->trainableVariables();
    //    $this->assertEquals(get_class($layerParams[0]),get_class($params[0]));
    //
    //    ob_start();
    //    $seq->summary();
    //    $dump = ob_get_clean();
    //    $summary =
    //    'Layer(type)                  Output Shape               Param #   '."\n".
    //    '=================================================================='."\n".
    //    'dense(Dense)                 (10)                       20        '."\n".
    //    'dense_1(Dense)               (10)                       20        '."\n".
    //    '=================================================================='."\n".
    //    'Weights                      Shape                      Param #   '."\n".
    //    '=================================================================='."\n".
    //    'No name                      (2)                        2         '."\n".
    //    'No name                      (2)                        2         '."\n".
    //    '=================================================================='."\n".
    //    'Total params: 44'."\n";
    //    $this->assertEquals($summary,$dump);
    //
    //}

    public function testCustomMetric()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,input_shape:[2],
                activation:'sigmoid'),
            $nn->layers()->Dense($units=2,
                activation:'softmax'),
        ]);

        $cateAcc = $nn->metrics->CategoricalAccuracy();
        $metricFn = function ($trues, $preds) use ($cateAcc) {
            $value = $cateAcc($trues, $preds);
            return $value;
        };

        $model->compile(
            loss:$nn->losses()->MeanSquaredError(),
            metrics:['loss'=>'loss','custom'=>$metricFn]
        );

        // training greater or less

        // training greater or less
        $x = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $t = $mo->array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]]);
        $v_x = $mo->array([[5, 1], [1, 5], [2, 6], [6, 1], [1, 7], [7, 2]]);
        $v_t = $mo->array([[0, 1], [1, 0], [1, 0], [0, 1], [1, 0], [0, 1]]);
        $history = $model->fit($x,$t,epochs:100,validation_data:[$v_x,$v_t],verbose:0);

        $y = $model->predict($x);
        $this->assertEquals($mo->argMax($t,axis:1)->toArray(),
                            $mo->argMax($y,axis:1)->toArray());

        $this->assertEquals(['loss','custom','val_loss','val_custom'],array_keys($history));

        if($this->plot) {
            $plt->plot($mo->array($history['loss']),null,null,'loss');
            $plt->plot($mo->array($history['val_loss']),null,null,'val_loss');
            $plt->plot($mo->array($history['custom']),null,null,'custom');
            $plt->plot($mo->array($history['val_custom']),null,null,'val_custom');
            $plt->legend();
            $plt->title('Custom Metric');
            $plt->show();
        }
    }

}
