<?php
namespace RindowTest\NeuralNetworks\Model\ModelLoaderTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\Math\Plot\Renderer\GDDriver;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Model\ModelLoader;
use PDO;

class Test extends TestCase
{
    private $plot=true;
    private $filename;

    public function getPlotConfig()
    {
        return [
            'renderer.skipCleaning' => true,
            'renderer.skipRunViewer' => getenv('TRAVIS_PHP_VERSION') ? true : false,
        ];
    }

    public function setUp() : void
    {
        $this->filename = __DIR__.'/../../../tmp/savedmodel.hda.sqlite3';
        $pdo = new PDO('sqlite:'.$this->filename);
        $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
        $sql = "DROP TABLE IF EXISTS hda";
        $stat = $pdo->exec($sql);
        unset($stat);
        unset($pdo);
    }

    public function testCleanUp()
    {
        $renderer = new GDDriver();
        $renderer->cleanUp();
        $this->assertTrue(true);
    }

    public function testModelFromConfig()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $loader = new ModelLoader($backend,$nn);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,['input_shape'=>[2]]),
            $nn->layers()->BatchNormalization(),
            $nn->layers()->Activation('sigmoid'),
            $nn->layers()->Dense($units=2,['activation'=>'softmax']),
        ]);
        $model->compile();
        $json = $model->toJson();
        $config = json_decode($json,true);


        // load model
        $model = $loader->modelFromConfig($config);
        $this->assertEquals($json,$model->toJson());

        $x = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $t = $mo->array([0, 0, 0, 1, 1, 1]);
        $history = $model->fit($x,$t,['epochs'=>100,'verbose'=>0]);

        $y = $model->predict($x);
        $this->assertEquals($t->toArray(),$mo->argMax($y,$axis=1)->toArray());
    }

    public function testSaveAndLoadModelDefault()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $nn = new NeuralNetworks($mo,$backend);

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

        $model->save($this->filename);

        // load model
        $model = $nn->models()->loadModel($this->filename);

        [$loss2,$accuracy2] = $model->evaluate($x,$t);
        $this->assertLessThan(0.5,abs($loss-$loss2));
        $this->assertLessThan(0.5,abs($accuracy-$accuracy2));
        //$y = $model->predict($x);
        //$this->assertEquals($t->toArray(),$mo->argMax($y,$axis=1)->toArray());
    }

    public function testSaveAndLoadModelPortable()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $plt = new Plot($this->getPlotConfig(),$mo);

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
        $y = $model->predict($x);
        [$loss,$accuracy] = $model->evaluate($x,$t);

        $model->save($this->filename,$portable=true);

        // load model
        $model = $nn->models()->loadModel($this->filename);

        $z = $model->predict($x);
        if($this->plot) {
            [$fig,$ax] = $plt->subplots(2);
            $diff = $mo->f('abs',$mo->select($mo->op($y,'-',$z),$mo->arange($t->size()),$mo->zeros([$t->size()])));
            $ax[0]->bar($mo->arange($diff->size()),$diff,null,null,'difference');
            $ax[0]->legend();
            $ax[1]->plot($mo->array($history['loss']),null,null,'loss');
            $ax[1]->plot($mo->array($history['accuracy']),null,null,'accuracy');
            $ax[1]->legend();
            $plt->title('save portable mode');
            $plt->show();
        }

        [$loss2,$accuracy2] = $model->evaluate($x,$t);
        $this->assertLessThan(0.5,abs($loss-$loss2));
        $this->assertLessThan(0.5,abs($accuracy-$accuracy2));
        //$this->assertEquals($t->toArray(),$mo->argMax($y,$axis=1)->toArray());
    }
}
