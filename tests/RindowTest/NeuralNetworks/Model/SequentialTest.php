<?php
namespace RindowTest\NeuralNetworks\Model\SequentialTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Model\ModelLoader;
use Rindow\Math\Plot\Plot;
use Rindow\Math\Plot\Renderer\GDDriver;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Callback\AbstractCallback;

class WeightLog extends AbstractCallback
{
    protected $gradlog = [];
    public function __construct($mo)
    {
        parent::__construct();
        $this->mo = $mo;
        $this->prev_w = null;
    }

    public function onEpochEnd(int $epoch, array $logs=null) : void
    {
        #echo "\n";
        $model = $this->getModel();
        $K = $model->backend();
        $weights = $model->weights();
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
            #echo sprintf("%7.2f",strval())." ";
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

class Test extends TestCase
{
    protected $plot = false;

    public function newBackend($mo)
    {
        $builder = new NeuralNetworks($mo);
        return $builder->backend();
    }

    public function getPlotConfig()
    {
        return [
            'renderer.skipCleaning' => true,
            'renderer.skipRunViewer' => getenv('TRAVIS_PHP_VERSION') ? true : false,
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
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense(
                $units=4,
                ['input_shape'=>[2],
                'activation'=>'sigmoid']),
            $nn->layers()->Dense(
                $units=3,
                ['activation'=>'softmax']),
        ]);

        $model->compile();
        $layers = $model->layers();
        $lossFunction = $model->lossFunction();
        $weights = $model->weights();
        $grads = $model->grads();

        $this->assertCount(2,$layers);
        $this->assertInstanceof(
            'Rindow\NeuralNetworks\Layer\Dense',$layers[0]);
        $this->assertInstanceof(
            'Rindow\NeuralNetworks\Activation\Sigmoid',$layers[0]->getActivation());
        $this->assertInstanceof(
            'Rindow\NeuralNetworks\Layer\Dense',$layers[1]);
        $this->assertInstanceof(
            'Rindow\NeuralNetworks\Loss\SparseCategoricalCrossEntropy',$layers[1]->getActivation());
        $this->assertInstanceof(
            'Rindow\NeuralNetworks\Loss\SparseCategoricalCrossEntropy',$lossFunction);
        $this->assertTrue($layers[1]->getActivation()->fromLogits());
        $this->assertTrue($lossFunction->fromLogits());

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

        $this->assertNotEquals($mo->zeros([2,4])->toArray(), $weights[0]->toArray());
        $this->assertEquals(   $mo->zeros([4])->toArray(),   $weights[1]->toArray());
        $this->assertNotEquals($mo->zeros([4,3])->toArray(), $weights[2]->toArray());
        $this->assertEquals(   $mo->zeros([3])->toArray(),   $weights[3]->toArray());

        $this->assertEquals($mo->zeros([2,4])->toArray(),    $grads[0]->toArray());
        $this->assertEquals($mo->zeros([4])->toArray(),      $grads[1]->toArray());
        $this->assertEquals($mo->zeros([4,3])->toArray(),    $grads[2]->toArray());
        $this->assertEquals($mo->zeros([3])->toArray(),      $grads[3]->toArray());
    }

    public function testFitAndPredictWithDefaults()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,['input_shape'=>[2],
                    'activation'=>'sigmoid']),
            $nn->layers()->Dense($units=2,['activation'=>'softmax']),
        ]);

        $model->compile();
        $this->assertTrue($model->layers()[1]->getActivation()->fromLogits());
        $this->assertTrue($model->lossFunction()->fromLogits());

        // training greater or less
        $x = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $t = $mo->array([0, 0, 0, 1, 1, 1]);
        $history = $model->fit($x,$t,['epochs'=>100,'verbose'=>0]);

        $y = $model->predict($x);
        $this->assertEquals($t->toArray(),$mo->argMax($y,$axis=1)->toArray());

        //$plt->plot($mo->array($history['loss']));
        //$plt->plot($mo->array($history['accuracy']));
        //$plt->title('fit and predict');
        //$plt->show();
    }

    public function testEvaluate()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,['input_shape'=>[2],
                'activation'=>'sigmoid']),
            $nn->layers()->Dense($units=2,['activation'=>'softmax']),
        ]);

        $model->compile();

        // training greater or less
        $x = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $t = $mo->array([0, 0, 0, 1, 1, 1]);
        $history = $model->fit($x,$t,['epochs'=>100,'verbose'=>0]);

        [$loss,$accuracy] = $model->evaluate($x,$t);
        $this->assertLessThan(1.0,$loss);
        $this->assertEquals(1.0,$accuracy);
    }

    public function testFitWithEvaluate()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,['input_shape'=>[2],
                'activation'=>'sigmoid']),
            $nn->layers()->Dense($units=2,
            ['activation'=>'softmax']),
        ]);

        $model->compile();

        // training greater or less
        $x = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $t = $mo->array([0, 0, 0, 1, 1, 1]);
        $v_x = $mo->array([[5, 1], [1, 5], [2, 6], [6, 1], [1, 7], [7, 2]]);
        $v_t = $mo->array([1, 0, 0, 1, 0, 1]);
        $history = $model->fit($x,$t,['epochs'=>100,'validation_data'=>[$v_x,$v_t],'verbose'=>0]);

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
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,[
                'input_shape'=>[2],
                'kernel_initializer'=>'he_normal',
                'activation'=>'relu']),
            $nn->layers()->Dense($units=2,
                 ['activation'=>'softmax']),
        ]);

        $model->compile([
            'optimizer'=>$nn->optimizers()->Adam()
        ]);
        $this->assertTrue($model->layers()[1]->getActivation()->fromLogits());
        $this->assertTrue($model->lossFunction()->fromLogits());

        // training greater or less
        $x = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $t = $mo->array([0, 0, 0, 1, 1, 1]);
        $v_x = $mo->array([[5, 1], [1, 5], [2, 6], [6, 1], [1, 7], [7, 2]]);
        $v_t = $mo->array([1, 0, 0, 1, 0, 1]);
        $history = $model->fit($x,$t,['epochs'=>100,'validation_data'=>[$v_x,$v_t],'verbose'=>0]);

        $y = $model->predict($x);
        $this->assertEquals($t->toArray(),$mo->argMax($y,$axis=1)->toArray());

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
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,['input_shape'=>[2],
                'activation'=>'sigmoid']),
            $nn->layers()->Dense($units=2,
            ['activation'=>'softmax']),
        ]);

        $model->compile([
            'loss'=>$nn->losses()->MeanSquaredError()
        ]);

        // training greater or less

        // training greater or less
        $x = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $t = $mo->array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]]);
        $v_x = $mo->array([[5, 1], [1, 5], [2, 6], [6, 1], [1, 7], [7, 2]]);
        $v_t = $mo->array([[0, 1], [1, 0], [1, 0], [0, 1], [1, 0], [0, 1]]);
        $history = $model->fit($x,$t,['epochs'=>100,'validation_data'=>[$v_x,$v_t],'verbose'=>0]);

        $y = $model->predict($x);
        $this->assertEquals($mo->argMax($t,$axis=1)->toArray(),
                            $mo->argMax($y,$axis=1)->toArray());

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
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,['input_shape'=>[2],
                'activation'=>'relu']),
            $nn->layers()->Dropout($rate=0.15),
            $nn->layers()->Dense($units=2,['activation'=>'softmax']),
        ]);

        $model->compile();

        // training greater or less
        $x = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $t = $mo->array([0, 0, 0, 1, 1, 1]);
        $v_x = $mo->array([[5, 1], [1, 5], [2, 6], [6, 1], [1, 7], [7, 2]]);
        $v_t = $mo->array([1, 0, 0, 1, 0, 1]);
        $history = $model->fit($x,$t,['epochs'=>100,'validation_data'=>[$v_x,$v_t],'verbose'=>0]);

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

    public function testFitWithBatchNorm()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,['input_shape'=>[2]]),
            $nn->layers()->BatchNormalization(),
            $nn->layers()->Activation('sigmoid'),
            $nn->layers()->Dense($units=2,
            ['activation'=>'softmax']),
        ]);

        $model->compile();

        // training greater or less
        $x = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $t = $mo->array([0, 0, 0, 1, 1, 1]);
        $v_x = $mo->array([[5, 1], [1, 5], [2, 6], [6, 1], [1, 7], [7, 2]]);
        $v_t = $mo->array([1, 0, 0, 1, 0, 1]);
        $history = $model->fit($x,$t,['epochs'=>100,'validation_data'=>[$v_x,$v_t],'verbose'=>0]);

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
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,['input_shape'=>[2],
                'activation'=>'sigmoid']),
            $nn->layers()->Dense($units=1,
            ['activation'=>'sigmoid']),
        ]);

        $model->compile([
            'loss'=>$nn->losses()->BinaryCrossEntropy(),
            #'optimizer'=>'adam',
        ]);
        //$model->summary();
        $this->assertTrue( $model->layers()[1]->getActivation()->fromLogits());
        $this->assertTrue( $model->lossFunction()->fromLogits());

        // training greater or less
        $x = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $t = $mo->array([0, 0, 0, 1, 1, 1]);
        $v_x = $mo->array([[5, 1], [1, 5], [2, 6], [6, 1], [1, 7], [7, 2]]);
        $v_t = $mo->array([1, 0, 0, 1, 0, 1]);
        $history = $model->fit($x,$t,['epochs'=>100,'batch_size'=>16,'validation_data'=>[$v_x,$v_t],'verbose'=>0]);

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
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,['input_shape'=>[2],
                'activation'=>'sigmoid']),
            $nn->layers()->Dense($units=2,
            ['activation'=>'softmax']),
        ]);

        $model->compile([
            'loss'=>$nn->losses()->CategoricalCrossEntropy(),
        ]);
        $this->assertTrue( $model->layers()[1]->getActivation()->fromLogits());
        $this->assertTrue( $model->lossFunction()->fromLogits());

        // training greater or less
        $x = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $t = $mo->array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]]);
        $v_x = $mo->array([[5, 1], [1, 5], [2, 6], [6, 1], [1, 7], [7, 2]]);
        $v_t = $mo->array([[0, 1], [1, 0], [1, 0], [0, 1], [1, 0], [0, 1]]);
        $history = $model->fit($x,$t,['epochs'=>100,'validation_data'=>[$v_x,$v_t],'verbose'=>0]);

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
        if(extension_loaded('rindow_openblas')) {
            $num_of_filters=128;
            $epoch = 300;
        } else {
            $num_of_filters=16;
            $epoch = 50;
        }
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Conv1D(
                $filters=$num_of_filters,#128,
                $kernel_size=3,
                ['input_shape'=>[10,1],
                'kernel_initializer'=>'he_normal',
                'activation'=>'relu']),
            $nn->layers()->MaxPooling1D(),
            $nn->layers()->Conv1D($num_of_filters/*128*/,3),
            $nn->layers()->Flatten(),
            $nn->layers()->Activation('softmax'),
        ]);

        $model->compile([
            'loss'=>$nn->losses()->SparseCategoricalCrossEntropy(),
            'optimizer'=>$nn->optimizers()->Adam(),
        ]);

        // training greater or less
        $x = $mo->array([
            [[0.1],[0.1],[0.2],[0.2],[0.3],[0.3],[0.4],[0.4],[0.5],[0.5]],
            [[0.9],[0.9],[0.8],[0.8],[0.7],[0.7],[0.6],[0.6],[0.5],[0.5]],
            [[0.5],[0.5],[0.6],[0.6],[0.7],[0.7],[0.8],[0.8],[0.9],[0.9]],
            [[0.5],[0.5],[0.4],[0.4],[0.3],[0.3],[0.2],[0.2],[0.1],[0.1]],
        ]);
        $t = $mo->array(
            [1,0,1,0]
        );
        $v_x = $mo->array([
            [[0.1],[0.1],[0.25],[0.25],[0.35],[0.35],[0.45],[0.45],[0.6], [0.6] ],
            [[0.9],[0.9],[0.7], [0.7], [0.5], [0.5], [0.3], [0.3], [0.1], [0.1] ],
            [[0.1],[0.1],[0.11],[0.11],[0.12],[0.12],[0.13],[0.13],[0.14],[0.14]],
            [[0.5],[0.5],[0.45],[0.45],[0.4], [0.4], [0.35],[0.35],[0.3], [0.3] ],
        ]);
        $v_t = $mo->array(
            [1,0,1,0]
        );
        $history = $model->fit(
            $x,$t,
            ['epochs'=>$epoch/*300*/,'validation_data'=>[$v_x,$v_t],'verbose'=>0]);

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
        if(extension_loaded('rindow_openblas')) {
            $num_of_filters=128;
            $epoch = 300;
        } else {
            $num_of_filters=8;
            $epoch = 30;
        }
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Conv2D(
                $filters=$num_of_filters,#128,
                $kernel_size=3,
                ['input_shape'=>[10,10,1],
                'kernel_initializer'=>'he_normal',
                'activation'=>'relu',
                #'activation'=>'softmax',
                ]),
            $nn->layers()->MaxPooling2D(),
            $nn->layers()->Conv2D($num_of_filters/*128*/,3),
            $nn->layers()->Flatten(),
            $nn->layers()->Activation('softmax'),
        ]);

        $model->compile([
            'loss'=>$nn->losses()->SparseCategoricalCrossEntropy(),
            'optimizer'=>$nn->optimizers()->Adam(),
        ]);

        // training greater or less
        $x = $mo->zeros([4,10,10,1]);
        for($i=0;$i<10;$i++) { $x[0][$i][$i][0]=1.0;}
        for($i=0;$i<10;$i++) { $x[1][$i][9-$i][0]=1.0;}
        for($i=1;$i<9;$i++)  { $x[2][$i+1][$i+1][0]=1.0;}
        for($i=1;$i<9;$i++)  { $x[3][$i+1][9-$i][0]=1.0;}
        $t = $mo->array(
            [1,0,1,0]
        );
        $v_x = $mo->zeros([4,10,10,1]);
        for($i=0;$i<8;$i++) { $x[0][$i][$i+2][0]=1.0;}
        for($i=0;$i<8;$i++) { $x[1][$i][9-$i][0]=1.0;}
        for($i=1;$i<8;$i++) { $x[2][$i+1][$i][0]=1.0;}
        for($i=1;$i<8;$i++) { $x[3][$i+2][9-$i][0]=1.0;}
        $v_t = $mo->array(
            [1,0,1,0]
        );
        $history = $model->fit(
            $x,$t,
            ['epochs'=>$epoch/*100*/,'validation_data'=>[$v_x,$v_t],'verbose'=>0]);

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
        if(extension_loaded('rindow_openblas')) {
            $num_of_filters=128;
            $epoch = 300;
        } else {
            $num_of_filters=8;
            $epoch = 20;
        }
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Conv3D(
                $filters=$num_of_filters,#128,
                $kernel_size=3,
                ['input_shape'=>[10,10,10,1],
                'kernel_initializer'=>'he_normal',
                'activation'=>'relu']),
            $nn->layers()->MaxPooling3D(),
            $nn->layers()->Conv3D($num_of_filters/*128*/,3),
            $nn->layers()->Flatten(),
            $nn->layers()->Activation('softmax'),
        ]);

        $model->compile([
            'loss'=>$nn->losses()->SparseCategoricalCrossEntropy(),
        ]);

        // training greater or less
        $x = $mo->zeros([4,10,10,10,1]);
        for($i=0;$i<10;$i++) { $x[0][$i][$i][$i][0]=1.0;}
        for($i=0;$i<10;$i++) { $x[1][$i][$i][9-$i][0]=1.0;}
        for($i=1;$i<9;$i++)  { $x[2][$i][$i+1][$i+1][0]=1.0;}
        for($i=1;$i<9;$i++)  { $x[3][$i][$i+1][9-$i][0]=1.0;}
        $t = $mo->array(
            [1,0,1,0]
        );
        $v_x = $mo->zeros([4,10,10,10,1]);
        for($i=0;$i<8;$i++) { $x[0][$i][$i][$i+2][0]=1.0;}
        for($i=0;$i<8;$i++) { $x[1][$i][$i][9-$i][0]=1.0;}
        for($i=1;$i<8;$i++) { $x[2][$i][$i+1][$i][0]=1.0;}
        for($i=1;$i<8;$i++) { $x[3][$i][$i+2][9-$i][0]=1.0;}
        $v_t = $mo->array(
            [1,0,1,0]
        );
        $history = $model->fit($x,$t,['epochs'=>$epoch/*100*/,'validation_data'=>[$v_x,$v_t],'verbose'=>0]);

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
        if(extension_loaded('rindow_openblas')) {
            $num_of_filters=128;
            $epoch = 300;
        } else {
            $num_of_filters=16;
            $epoch = 50;
        }
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Conv1D(
                $filters=$num_of_filters,#128,
                $kernel_size=3,
                ['input_shape'=>[10,1],
                'kernel_initializer'=>'he_normal',
                'activation'=>'relu']),
            $nn->layers()->AveragePooling1D(),
            $nn->layers()->Conv1D($num_of_filters/*128*/,3),
            $nn->layers()->Flatten(),
            $nn->layers()->Activation('softmax'),
        ]);

        $model->compile([
            'loss'=>$nn->losses()->SparseCategoricalCrossEntropy(),
            'optimizer'=>$nn->optimizers()->Adam(),
        ]);

        // training greater or less
        $x = $mo->array([
            [[0.1],[0.1],[0.2],[0.2],[0.3],[0.3],[0.4],[0.4],[0.5],[0.5]],
            [[0.9],[0.9],[0.8],[0.8],[0.7],[0.7],[0.6],[0.6],[0.5],[0.5]],
            [[0.5],[0.5],[0.6],[0.6],[0.7],[0.7],[0.8],[0.8],[0.9],[0.9]],
            [[0.5],[0.5],[0.4],[0.4],[0.3],[0.3],[0.2],[0.2],[0.1],[0.1]],
        ]);
        $t = $mo->array(
            [1,0,1,0]
        );
        $v_x = $mo->array([
            [[0.1],[0.1],[0.25],[0.25],[0.35],[0.35],[0.45],[0.45],[0.6], [0.6] ],
            [[0.9],[0.9],[0.7], [0.7], [0.5], [0.5], [0.3], [0.3], [0.1], [0.1] ],
            [[0.1],[0.1],[0.11],[0.11],[0.12],[0.12],[0.13],[0.13],[0.14],[0.14]],
            [[0.5],[0.5],[0.45],[0.45],[0.4], [0.4], [0.35],[0.35],[0.3], [0.3] ],
        ]);
        $v_t = $mo->array(
            [1,0,1,0]
        );
        $history = $model->fit(
            $x,$t,
            ['epochs'=>$epoch/*300*/,'validation_data'=>[$v_x,$v_t],'verbose'=>0]);

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
        if(extension_loaded('rindow_openblas')) {
            $num_of_filters=128;
            $epoch = 300;
        } else {
            $num_of_filters=8;
            $epoch = 30;
        }
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Conv2D(
                $filters=$num_of_filters,#128,
                $kernel_size=3,
                ['input_shape'=>[10,10,1],
                'kernel_initializer'=>'he_normal',
                'activation'=>'relu',
                #'activation'=>'softmax',
                ]),
            $nn->layers()->AveragePooling2D(),
            $nn->layers()->Conv2D($num_of_filters/*128*/,3),
            $nn->layers()->Flatten(),
            $nn->layers()->Activation('softmax'),
        ]);

        $model->compile([
            'loss'=>$nn->losses()->SparseCategoricalCrossEntropy(),
            'optimizer'=>$nn->optimizers()->Adam(),
        ]);

        // training greater or less
        $x = $mo->zeros([4,10,10,1]);
        for($i=0;$i<10;$i++) { $x[0][$i][$i][0]=1.0;}
        for($i=0;$i<10;$i++) { $x[1][$i][9-$i][0]=1.0;}
        for($i=1;$i<9;$i++)  { $x[2][$i+1][$i+1][0]=1.0;}
        for($i=1;$i<9;$i++)  { $x[3][$i+1][9-$i][0]=1.0;}
        $t = $mo->array(
            [1,0,1,0]
        );
        $v_x = $mo->zeros([4,10,10,1]);
        for($i=0;$i<8;$i++) { $x[0][$i][$i+2][0]=1.0;}
        for($i=0;$i<8;$i++) { $x[1][$i][9-$i][0]=1.0;}
        for($i=1;$i<8;$i++) { $x[2][$i+1][$i][0]=1.0;}
        for($i=1;$i<8;$i++) { $x[3][$i+2][9-$i][0]=1.0;}
        $v_t = $mo->array(
            [1,0,1,0]
        );
        $history = $model->fit(
            $x,$t,
            ['epochs'=>$epoch,'validation_data'=>[$v_x,$v_t],'verbose'=>0]);

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
        if(extension_loaded('rindow_openblas')) {
            $num_of_filters=128;
            $epoch = 300;
        } else {
            $num_of_filters=8;
            $epoch = 20;
        }
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Conv3D(
                $filters=$num_of_filters,#128,
                $kernel_size=3,
                ['input_shape'=>[10,10,10,1],
                'kernel_initializer'=>'he_normal',
                'activation'=>'relu']),
            $nn->layers()->AveragePooling3D(),
            $nn->layers()->Conv3D($num_of_filters/*128*/,3),
            $nn->layers()->Flatten(),
            $nn->layers()->Activation('softmax'),
        ]);

        $model->compile([
            'loss'=>$nn->losses()->SparseCategoricalCrossEntropy(),
        ]);

        // training greater or less
        $x = $mo->zeros([4,10,10,10,1]);
        for($i=0;$i<10;$i++) { $x[0][$i][$i][$i][0]=1.0;}
        for($i=0;$i<10;$i++) { $x[1][$i][$i][9-$i][0]=1.0;}
        for($i=1;$i<9;$i++)  { $x[2][$i][$i+1][$i+1][0]=1.0;}
        for($i=1;$i<9;$i++)  { $x[3][$i][$i+1][9-$i][0]=1.0;}
        $t = $mo->array(
            [1,0,1,0]
        );
        $v_x = $mo->zeros([4,10,10,10,1]);
        for($i=0;$i<8;$i++) { $x[0][$i][$i][$i+2][0]=1.0;}
        for($i=0;$i<8;$i++) { $x[1][$i][$i][9-$i][0]=1.0;}
        for($i=1;$i<8;$i++) { $x[2][$i][$i+1][$i][0]=1.0;}
        for($i=1;$i<8;$i++) { $x[3][$i][$i+2][9-$i][0]=1.0;}
        $v_t = $mo->array(
            [1,0,1,0]
        );
        $history = $model->fit($x,$t,['epochs'=>$epoch/*100*/,'validation_data'=>[$v_x,$v_t],'verbose'=>0]);

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

    public function testFitEmbeding()
    {
        if(extension_loaded('rindow_openblas')) {
            $epoch = 300;
        } else {
            $epoch = 50;
        }
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Embedding($inputDim=10,$outputDim=10,
                [
                    'input_length'=>4,
                    #'kernel_initializer'=>'glorot_normal',
            ]),
            $nn->layers()->Activation('softmax'),
        ]);

        $model->compile([
            'loss'=>$nn->losses()->SparseCategoricalCrossEntropy(),
            'optimizer'=>'adam',
        ]);
        //$model->summary();
        //foreach($model->weights() as $w) {
        //    echo '('.implode(',',$w->shape()).'):'.$mo->la()->amax($w)."\n";
        //}

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
        ]);
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
        ]);

        $history = $model->fit($x,$t,['epochs'=>$epoch/*300*/,'batch_size'=>1,'validation_data'=>[$v_x,$v_t],'verbose'=>0]);

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
        if(extension_loaded('rindow_openblas')) {
            $epoch = 300;
        } else {
            $epoch = 100;
        }
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->SimpleRNN(
                $units=16,
                [
                    'input_shape'=>[4,10],
                    #'input_shape'=>[1,10],
                    #'kernel_initializer'=>'glorot_normal',
                    #'recurrent_initializer'=>'glorot_normal',
                    #'recurrent_initializer'=>'glorot_normal',
            ]),
            $nn->layers()->Activation('softmax'),
        ]);

        $model->compile([
            'loss'=>$nn->losses()->SparseCategoricalCrossEntropy(),
            'optimizer'=>'adam',
        ]);
        #$model->summary();
        #foreach($model->weights() as $w) {
        #    echo '('.implode(',',$w->shape()).'):'.$mo->la()->amax($w)."\n";
        #}

        // training up and down
        $x = $mo->array([
            [0,1,2,9],
            [9,8,7,6],
            [1,3,3,4],
            [5,4,3,2],
        ]);
        $t = $mo->array(
            [1,0,1,0]
        );
        $v_x = $mo->array([
            [2,3,3,4],
            [1,1,1,4],
            [4,3,3,1],
            [9,3,3,2],
        ]);
        $v_t = $mo->array(
            [1,1,0,0]
        );
        $x = $mo->la()->onehot($x->reshape([16]),$numClass=10)->reshape([4,4,10]);
        $v_x = $mo->la()->onehot($v_x->reshape([16]),$numClass=10)->reshape([4,4,10]);

        $history = $model->fit($x,$t,['epochs'=>$epoch/*300*/,'batch_size'=>1,'validation_data'=>[$v_x,$v_t],'verbose'=>0]);

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
        if(extension_loaded('rindow_openblas')) {
            $epoch = 300;
        } else {
            $epoch = 100;
        }
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->SimpleRNN(
                $units=10,
                [
                    'input_shape'=>[4,10],
                    #'input_shape'=>[1,10],
                    #'kernel_initializer'=>'glorot_normal',
                    #'recurrent_initializer'=>'glorot_normal',
                    #'recurrent_initializer'=>'glorot_normal',
                    'return_sequences'=>true,
            ]),
            $nn->layers()->Activation('softmax'),
        ]);

        $model->compile([
            'loss'=>$nn->losses()->SparseCategoricalCrossEntropy(),
            'optimizer'=>'adam',
        ]);
        #$model->summary();
        #foreach($model->weights() as $w) {
        #    echo '('.implode(',',$w->shape()).'):'.$mo->la()->amax($w)."\n";
        #}

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
        ]);
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
        ]);
        $x = $mo->la()->onehot($x->reshape([16]),$numClass=10)->reshape([4,4,10]);
        $v_x = $mo->la()->onehot($v_x->reshape([16]),$numClass=10)->reshape([4,4,10]);

        $callback = new WeightLog($mo);
        $history = $model->fit($x,$t,['epochs'=>$epoch/*300*/,'batch_size'=>1,'validation_data'=>[$v_x,$v_t],'verbose'=>0,'callbacks'=>[$callback]]);

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
        if(extension_loaded('rindow_openblas')) {
            $epoch = 300;
        } else {
            $epoch = 100;
        }
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense(
                $units=16,
                [
                    #'input_shape'=>[4,10],
                    'input_shape'=>[10],
                    #'kernel_initializer'=>'glorot_normal',
                    #'recurrent_initializer'=>'glorot_normal',
                    #'recurrent_initializer'=>'glorot_normal',
                    #'activation'=>'sigmoid',
                    'activation'=>'tanh',
            ]),
            $nn->layers()->Activation('softmax'),
        ]);

        $model->compile([
            'loss'=>'sparse_categorical_crossentropy',
            'optimizer'=>'adam',
            #'optimizer'=>'sgd',
        ]);
        #$model->summary();
        #foreach($model->weights() as $w) {
        #    echo '('.implode(',',$w->shape()).'):'.$mo->la()->amax($w)."\n";
        #}
        #return;

        $x = $mo->array([
            [0],
            [9],
            [1],
            [5],
        ]);
        $t = $mo->array(
            [0,9,1,5]
        );
        $v_x = $mo->array([
            [2],
            [1],
            [4],
            [9],
        ]);
        $v_t = $mo->array(
            [2,1,4,9]
        );
        $x = $mo->la()->onehot($x->reshape([4]),$numClass=10)->reshape([4,10]);
        $v_x = $mo->la()->onehot($v_x->reshape([4]),$numClass=10)->reshape([4,10]);

        $history = $model->fit($x,$t,['epochs'=>$epoch/*300*/,'batch_size'=>1,'validation_data'=>[$v_x,$v_t],'verbose'=>0]);

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
        if(extension_loaded('rindow_openblas')) {
            $epoch = 300;
        } else {
            $epoch = 100;
        }
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->LSTM(
                $units=16,
                [
                    'input_shape'=>[4,10],
                ]),
                $nn->layers()->Activation('softmax'),
        ]);

        $model->compile([
            'loss'=>$nn->losses()->SparseCategoricalCrossEntropy(),
            'optimizer'=>$nn->optimizers()->Adam(),
        ]);
        //$model->summary();

        // training up and down
        $x = $mo->array([
            [0,1,2,9],
            [9,8,7,6],
            [1,3,3,4],
            [5,4,3,2],
        ]);
        $t = $mo->array(
            [1,0,1,0]
        );
        $v_x = $mo->array([
            [2,3,3,4],
            [1,1,1,4],
            [4,3,3,1],
            [9,3,3,2],
        ]);
        $v_t = $mo->array(
            [1,1,0,0]
        );
        $x = $mo->la()->onehot($x->reshape([16]),$numClass=10)->reshape([4,4,10]);
        $v_x = $mo->la()->onehot($v_x->reshape([16]),$numClass=10)->reshape([4,4,10]);
        $history = $model->fit($x,$t,['epochs'=>$epoch/*300*/,'batch_size'=>1,'validation_data'=>[$v_x,$v_t],'verbose'=>0]);

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
        if(extension_loaded('rindow_openblas')) {
            $epoch = 300;
        } else {
            $epoch = 100;
        }
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->LSTM(
                $units=16,
                [
                    'input_shape'=>[4,10],
                    'return_sequences'=>true,
                ]),
                $nn->layers()->Activation('softmax'),
        ]);

        $model->compile([
            'loss'=>$nn->losses()->SparseCategoricalCrossEntropy(),
            'optimizer'=>$nn->optimizers()->Adam(),
        ]);
        //$model->summary();

        // training up and down
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
        ]);
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
        ]);
        $x = $mo->la()->onehot($x->reshape([16]),$numClass=10)->reshape([4,4,10]);
        $v_x = $mo->la()->onehot($v_x->reshape([16]),$numClass=10)->reshape([4,4,10]);
        $history = $model->fit($x,$t,['epochs'=>$epoch/*300*/,'batch_size'=>1,'validation_data'=>[$v_x,$v_t],'verbose'=>0]);

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
        if(extension_loaded('rindow_openblas')) {
            $epoch = 300;
        } else {
            $epoch = 100;
        }
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->GRU(
                $units=16,
                [
                    'input_shape'=>[4,10],
                ]),
                $nn->layers()->Activation('softmax'),
        ]);

        $model->compile([
            'loss'=>$nn->losses()->SparseCategoricalCrossEntropy(),
            'optimizer'=>'adam',
        ]);
        #$model->summary();
        #foreach($model->weights() as $w) {
        #    echo '('.implode(',',$w->shape()).'):'.$mo->la()->amax($w)."\n";
        #}

        // training up and down
        $x = $mo->array([
            [0,1,2,9],
            [9,8,7,6],
            [1,3,3,4],
            [5,4,3,2],
        ]);
        $t = $mo->array(
            [1,0,1,0]
        );
        $v_x = $mo->array([
            [2,3,3,4],
            [1,1,1,4],
            [4,3,3,1],
            [9,3,3,2],
        ]);
        $v_t = $mo->array(
            [1,1,0,0]
        );
        $x = $mo->la()->onehot($x->reshape([16]),$numClass=10)->reshape([4,4,10]);
        $v_x = $mo->la()->onehot($v_x->reshape([16]),$numClass=10)->reshape([4,4,10]);
        $history = $model->fit($x,$t,['epochs'=>$epoch/*300*/,'batch_size'=>1,'validation_data'=>[$v_x,$v_t],'verbose'=>0]);

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
        if(extension_loaded('rindow_openblas')) {
            $epoch = 300;
        } else {
            $epoch = 100;
        }
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->GRU(
                $units=16,
                [
                    'input_shape'=>[4,10],
                    'reset_after'=>false,
                ]),
                $nn->layers()->Activation('softmax'),
        ]);

        $model->compile([
            'loss'=>$nn->losses()->SparseCategoricalCrossEntropy(),
            'optimizer'=>'adam',
        ]);
        #$model->summary();
        #foreach($model->weights() as $w) {
        #    echo '('.implode(',',$w->shape()).'):'.$mo->la()->amax($w)."\n";
        #}

        // training up and down
        $x = $mo->array([
            [0,1,2,9],
            [9,8,7,6],
            [1,3,3,4],
            [5,4,3,2],
        ]);
        $t = $mo->array(
            [1,0,1,0]
        );
        $v_x = $mo->array([
            [2,3,3,4],
            [1,1,1,4],
            [4,3,3,1],
            [9,3,3,2],
        ]);
        $v_t = $mo->array(
            [1,1,0,0]
        );
        $x = $mo->la()->onehot($x->reshape([16]),$numClass=10)->reshape([4,4,10]);
        $v_x = $mo->la()->onehot($v_x->reshape([16]),$numClass=10)->reshape([4,4,10]);
        $history = $model->fit($x,$t,['epochs'=>$epoch/*300*/,'batch_size'=>1,'validation_data'=>[$v_x,$v_t],'verbose'=>0]);

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
        if(extension_loaded('rindow_openblas')) {
            $epoch = 300;
        } else {
            $epoch = 100;
        }
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->GRU(
                $units=10,
                [
                    'input_shape'=>[4,10],
                    #'input_shape'=>[1,10],
                    #'kernel_initializer'=>'glorot_normal',
                    #'recurrent_initializer'=>'glorot_normal',
                    #'recurrent_initializer'=>'glorot_normal',
                    'return_sequences'=>true,
            ]),
            $nn->layers()->Activation('softmax'),
        ]);

        $model->compile([
            'loss'=>$nn->losses()->SparseCategoricalCrossEntropy(),
            'optimizer'=>'adam',
        ]);
        #$model->summary();
        #foreach($model->weights() as $w) {
        #    echo '('.implode(',',$w->shape()).'):'.$mo->la()->amax($w)."\n";
        #}

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
        ]);
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
        ]);
        $x = $mo->la()->onehot($x->reshape([16]),$numClass=10)->reshape([4,4,10]);
        $v_x = $mo->la()->onehot($v_x->reshape([16]),$numClass=10)->reshape([4,4,10]);

        $callback = new WeightLog($mo);
        $history = $model->fit($x,$t,['epochs'=>$epoch/*300*/,'batch_size'=>1,'validation_data'=>[$v_x,$v_t],'verbose'=>0,'callbacks'=>[$callback]]);

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
        if(extension_loaded('rindow_openblas')) {
            $epoch = 300;
        } else {
            $epoch = 50;
        }
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense(
                $units=16,
                [
                    'input_shape'=>[4,10],
                ]),
                $nn->layers()->Flatten(),
                $nn->layers()->RepeatVector(3),
                $nn->layers()->Dense(10),
                $nn->layers()->Activation('softmax'),
        ]);

        $model->compile([
            'loss'=>$nn->losses()->SparseCategoricalCrossEntropy(),
            'optimizer'=>'adam',
        ]);
        #$model->summary();
        #foreach($model->weights() as $w) {
        #    echo '('.implode(',',$w->shape()).'):'.$mo->la()->amax($w)."\n";
        #}

        // training up and down
        $x = $mo->array([
            [0,1,2,9],
            [9,8,7,6],
            [1,3,3,4],
            [5,4,3,2],
        ]);
        $t = $mo->array([
            [0,1,0],
            [1,0,1],
            [0,1,0],
            [1,0,1],
        ]);
        $v_x = $mo->array([
            [2,3,3,4],
            [1,1,1,4],
            [4,3,3,1],
            [9,3,3,2],
        ]);
        $v_t = $mo->array([
            [0,1,0],
            [0,1,0],
            [1,0,1],
            [1,0,1],
        ]);
        $x = $mo->la()->onehot($x->reshape([16]),$numClass=10)->reshape([4,4,10]);
        $v_x = $mo->la()->onehot($v_x->reshape([16]),$numClass=10)->reshape([4,4,10]);
        $history = $model->fit($x,$t,['epochs'=>$epoch/*300*/,'batch_size'=>1,'validation_data'=>[$v_x,$v_t],'verbose'=>0]);

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
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,['input_shape'=>[2]]),
            $nn->layers()->BatchNormalization(),
            $nn->layers()->Activation('sigmoid'),
            $nn->layers()->Dense($units=2,['activation'=>'softmax']),
        ]);
        $model->compile();
        $json = $model->toJson();
        $this->assertTrue(true);
    }

    public function testSaveAndLoadWeightsNormal()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $loader = new ModelLoader($backend,$nn);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,['input_shape'=>[2]]),
            $nn->layers()->BatchNormalization(),
            $nn->layers()->Activation('sigmoid'),
            $nn->layers()->Dense($units=2,['activation'=>'softmax']),
        ]);
        $model->compile();
        $x = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $t = $mo->array([0, 0, 0, 1, 1, 1]);
        $history = $model->fit($x,$t,['epochs'=>100,'verbose'=>0]);
        [$loss,$accuracy] = $model->evaluate($x,$t);
        //$y = $model->predict($x);
        //$this->assertEquals($t->toArray(),$mo->argMax($y,$axis=1)->toArray());

        // save config and weights
        $json = $model->toJson();
        $weights = [];
        $model->saveWeights($weights);
        $config = json_decode($json,true);

        // new model from config and load weights
        $model = $loader->modelFromConfig($config);
        $model->loadWeights($weights);

        [$loss2,$accuracy2] = $model->evaluate($x,$t);
        $this->assertLessThan(0.5,abs($loss-$loss2));
        $this->assertLessThan(0.5,abs($accuracy-$accuracy2));
        //$y = $model->predict($x);
        //$this->assertEquals($t->toArray(),$mo->argMax($y,$axis=1)->toArray());

    }
}
