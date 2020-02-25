<?php
namespace RindowTest\NeuralNetworks\Model\SequentialTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Model\ModelLoader;
use Rindow\Math\Plot\Plot;
use Rindow\Math\Plot\Renderer\GDDriver;

class Test extends TestCase
{
    protected $plot = false;

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
        $backend = new Backend($mo);
        $nn = new NeuralNetworks($mo,$backend);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=4,['input_shape'=>[2]]),
            $nn->layers()->Sigmoid(),
            $nn->layers()->Dense($units=3),
            $nn->layers()->Softmax(),
        ]);

        $model->compile();
        $layers = $model->layers();
        $lossFunction = $model->lossFunction();
        $weights = $model->weights();
        $grads = $model->grads();

        $this->assertCount(4,$layers);
        $this->assertInstanceof(
            'Rindow\NeuralNetworks\Layer\Dense',$layers[0]);
        $this->assertInstanceof(
            'Rindow\NeuralNetworks\Layer\Sigmoid',$layers[1]);
        $this->assertInstanceof(
            'Rindow\NeuralNetworks\Layer\Dense',$layers[2]);
        $this->assertInstanceof(
            'Rindow\NeuralNetworks\Loss\SparseCategoricalCrossEntropy',$layers[3]);
        $this->assertInstanceof(
            'Rindow\NeuralNetworks\Loss\SparseCategoricalCrossEntropy',$lossFunction);
        $this->assertTrue($layers[3]->fromLogits());
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
        $backend = new Backend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,['input_shape'=>[2]]),
            $nn->layers()->Sigmoid(),
            $nn->layers()->Dense($units=2),
            $nn->layers()->Softmax(),
        ]);

        $model->compile();
        $this->assertTrue($model->layers()[3]->fromLogits());
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
        $backend = new Backend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,['input_shape'=>[2]]),
            $nn->layers()->Sigmoid(),
            $nn->layers()->Dense($units=2),
            $nn->layers()->Softmax(),
        ]);

        $model->compile();

        // training greater or less
        $x = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $t = $mo->array([0, 0, 0, 1, 1, 1]);
        $history = $model->fit($x,$t,['epochs'=>100,'verbose'=>0]);

        [$loss,$accuracy] = $model->evaluate($x,$t);
        $this->assertLessThan(0.3,$loss);
        $this->assertEquals(1.0,$accuracy);
    }

    public function testFitWithEvaluate()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,['input_shape'=>[2]]),
            $nn->layers()->ReLU(),
            $nn->layers()->Dense($units=2),
            $nn->layers()->Softmax(),
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
        $backend = new Backend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,[
                'input_shape'=>[2],'kernel_initializer'=>'relu_normal']),
            $nn->layers()->ReLU(),
            $nn->layers()->Dense($units=2),
            $nn->layers()->Softmax(),
        ]);

        $model->compile([
            'optimizer'=>$nn->optimizers()->Adam()
        ]);
        $this->assertTrue($model->layers()[3]->fromLogits());
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
        $backend = new Backend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,[
                'input_shape'=>[2]]),
            $nn->layers()->Sigmoid(),
            $nn->layers()->Dense($units=2),
            $nn->layers()->Softmax(),
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
        $backend = new Backend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,['input_shape'=>[2]]),
            $nn->layers()->ReLU(),
            $nn->layers()->Dropout($rate=0.15),
            $nn->layers()->Dense($units=2),
            $nn->layers()->Softmax(),
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
        $backend = new Backend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,['input_shape'=>[2]]),
            $nn->layers()->BatchNormalization(),
            $nn->layers()->Sigmoid(),
            $nn->layers()->Dense($units=2),
            $nn->layers()->Softmax(),
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
        $backend = new Backend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,['input_shape'=>[2]]),
            $nn->layers()->Sigmoid(),
            $nn->layers()->Dense($units=1),
            $nn->layers()->Sigmoid(),
        ]);

        $model->compile([
            'loss'=>$nn->losses()->BinaryCrossEntropy(),
        ]);
        $this->assertTrue( $model->layers()[3]->fromLogits());
        $this->assertTrue( $model->lossFunction()->fromLogits());

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
            $plt->title('BinaryClassification');
            $plt->show();
        }
    }

    public function testFitOnehotCategoricalClassification()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,['input_shape'=>[2]]),
            $nn->layers()->Sigmoid(),
            $nn->layers()->Dense($units=2),
            $nn->layers()->Softmax(),
        ]);

        $model->compile([
            'loss'=>$nn->losses()->CategoricalCrossEntropy(),
        ]);
        $this->assertTrue( $model->layers()[3]->fromLogits());
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

    public function testToJson()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,['input_shape'=>[2]]),
            $nn->layers()->BatchNormalization(),
            $nn->layers()->Sigmoid(),
            $nn->layers()->Dense($units=2),
            $nn->layers()->Softmax(),
        ]);
        $model->compile();
        $json = $model->toJson();
        $this->assertTrue(true);
    }

    public function testSaveAndLoadWeightsNormal()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $loader = new ModelLoader($backend,$nn);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,['input_shape'=>[2]]),
            $nn->layers()->BatchNormalization(),
            $nn->layers()->Sigmoid(),
            $nn->layers()->Dense($units=2),
            $nn->layers()->Softmax(),
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
        $this->assertLessThan(0.3,abs($loss-$loss2));
        $this->assertLessThan(0.3,abs($accuracy-$accuracy2));
        //$y = $model->predict($x);
        //$this->assertEquals($t->toArray(),$mo->argMax($y,$axis=1)->toArray());

    }
}
