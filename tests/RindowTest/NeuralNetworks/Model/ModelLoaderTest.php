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
use Interop\Polite\Math\Matrix\NDArray;

class ModelLoaderTest extends TestCase
{
    private $plot=true;
    private $filename;

    public function newBackend($mo)
    {
        $builder = new NeuralNetworks($mo);
        return $builder->backend();
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
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $loader = new ModelLoader($nn);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,input_shape:[2]),
            $nn->layers()->BatchNormalization(),
            $nn->layers()->Activation('sigmoid'),
            $nn->layers()->Dense($units=2, activation:'softmax'),
        ]);
        $model->compile();
        $json = $model->toJson();
        $config = json_decode($json,true);


        // load model
        $model = $loader->modelFromConfig($config);
        $this->assertEquals($json,$model->toJson());

        $x = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $t = $mo->array([0, 0, 0, 1, 1, 1],dtype:NDArray::int32);
        $history = $model->fit($x,$t, epochs:100, verbose:0);

        $y = $model->predict($x);
        $this->assertEquals($t->toArray(),$mo->argMax($y,axis:1)->toArray());
    }

    public function testSaveAndLoadModelDefaultDenseBatchNrm()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,input_shape:[2]),
            $nn->layers()->BatchNormalization(),
            $nn->layers()->Activation('sigmoid'),
            $nn->layers()->Dense($units=2, activation:'softmax'),
        ]);
        $model->compile();
        $x = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $t = $mo->array([0, 0, 0, 1, 1, 1],dtype:NDArray::int32);
        $history = $model->fit($x,$t,epochs:100, verbose:0);
        $evals = $model->evaluate($x,$t);
        $y = $model->predict($x);

        $model->save($this->filename);

        // load model
        $model = $nn->models()->loadModel($this->filename);

        $evals2 = $model->evaluate($x,$t);
        $this->assertLessThan(0.5,abs($evals['loss']-$evals2['loss']));
        $this->assertLessThan(0.5,abs($evals['accuracy']-$evals2['accuracy']));
        $y2 = $model->predict($x);
        $this->assertLessThan(1e-7,$mo->la()->sum($mo->la()->square($mo->op($y,'-',$y2))));
        //$this->assertEquals($t->toArray(),$mo->argMax($y,axis:1)->toArray());
    }

    public function testSaveAndLoadModelDefaultRnnEmbed()
    {
        $mo = new MatrixOperator();
        $backend = $K = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $K = $nn->backend();

        $REVERSE = True;
        $WORD_VECTOR = 16;
        if(!$mo->isAdvanced() || getenv('BLOBDATA_TEST_SKIP')) {
            $UNITS = 16;
        } else {
            $UNITS = 128;
        }
        $question = $mo->array([
            [1,2,3,4,5,6],
            [3,4,5,6,7,8],
            [6,5,4,3,2,1],
            [8,7,6,5,4,3],
        ],NDArray::int32);
        $answer = $mo->array([
            [2,4,6],
            [4,6,8],
            [5,3,1],
            [7,5,3],
        ],NDArray::int32);
        $input_length = $question->shape()[1];
        $input_dict_size = $mo->max($question)+1;
        $output_length = $answer->shape()[1];
        $target_dict_size = $mo->max($answer)+1;

        $model = $nn->models()->Sequential([
            $nn->layers()->Embedding($input_dict_size, $WORD_VECTOR,
                input_length:$input_length
            ),
            # Encoder
            $nn->layers()->GRU($UNITS,
                go_backwards:$REVERSE,
                #reset_after:false,
            ),
            # Expand to answer length and peeking hidden states
            $nn->layers()->RepeatVector($output_length),
            # Decoder
            $nn->layers()->GRU($UNITS,
                return_sequences:true,
                go_backwards:$REVERSE,
                #reset_after:false,
            ),
            # Output
            $nn->layers()->Dense(
                $target_dict_size,
                activation:'softmax'
            ),
        ]);
        $model->compile(
            loss:'sparse_categorical_crossentropy',
            optimizer:'adam',
        );
        $history = $model->fit($question,$answer,epochs:10, verbose:0);
        $evals = $model->evaluate($question,$answer);
        $y = $model->predict($question);
        $layers = $model->layers();
        $embvals = $layers[0]->getParams();
        $gruvals = $layers[1]->getParams();
        $gru2vals = $layers[3]->getParams();
        $densevals = $layers[4]->getParams();

        $model->save($this->filename);

        // load model
        $model = $nn->models()->loadModel($this->filename);

        $evals2 = $model->evaluate($question,$answer);
        $this->assertLessThan(0.5,abs($evals['loss']-$evals2['loss']));
        $this->assertLessThan(0.5,abs($evals['accuracy']-$evals2['accuracy']));

        $layers1 = $model->layers();
        $embvals1 = $layers1[0]->getParams();
        $gruvals1 = $layers1[1]->getParams();
        $gru2vals1 = $layers1[3]->getParams();
        $densevals1 = $layers1[4]->getParams();

        $y2 = $model->predict($question);
        $layers2 = $model->layers();
        $embvals2 = $layers2[0]->getParams();
        $gruvals2 = $layers2[1]->getParams();
        $gru2vals2 = $layers2[3]->getParams();
        $densevals2 = $layers2[4]->getParams();
        $this->assertLessThan(1e-7,$mo->la()->sum($mo->la()->square($mo->op($K->NDArray($embvals[0]),'-',$K->NDArray($embvals2[0])))));
        $this->assertLessThan(1e-7,$mo->la()->sum($mo->la()->square($mo->op($K->NDArray($gruvals[0]),'-',$K->NDArray($gruvals2[0])))));
        $this->assertLessThan(1e-7,$mo->la()->sum($mo->la()->square($mo->op($K->NDArray($gruvals[1]),'-',$K->NDArray($gruvals2[1])))));
        $this->assertLessThan(1e-7,$mo->la()->sum($mo->la()->square($mo->op($K->NDArray($gruvals[2]),'-',$K->NDArray($gruvals2[2])))));
        $this->assertLessThan(1e-7,$mo->la()->sum($mo->la()->square($mo->op($K->NDArray($gru2vals[0]),'-',$K->NDArray($gru2vals2[0])))));
        $this->assertLessThan(1e-7,$mo->la()->sum($mo->la()->square($mo->op($K->NDArray($gru2vals[1]),'-',$K->NDArray($gru2vals2[1])))));
        $this->assertLessThan(1e-7,$mo->la()->sum($mo->la()->square($mo->op($K->NDArray($gru2vals[2]),'-',$K->NDArray($gru2vals2[2])))));
        $this->assertLessThan(1e-7,$mo->la()->sum($mo->la()->square($mo->op($K->NDArray($densevals[0]),'-',$K->NDArray($densevals2[0])))));
        $this->assertLessThan(1e-7,$mo->la()->sum($mo->la()->square($mo->op($K->NDArray($densevals[1]),'-',$K->NDArray($densevals2[1])))));

        $this->assertNotEquals(spl_object_id($embvals[0]),spl_object_id($embvals2[0]));
        $this->assertNotEquals(spl_object_id($gruvals[0]),spl_object_id($gruvals2[0]));
        $this->assertNotEquals(spl_object_id($gruvals[1]),spl_object_id($gruvals2[1]));
        $this->assertNotEquals(spl_object_id($gruvals[2]),spl_object_id($gruvals2[2]));
        $this->assertNotEquals(spl_object_id($gru2vals[0]),spl_object_id($gru2vals2[0]));
        $this->assertNotEquals(spl_object_id($gru2vals[1]),spl_object_id($gru2vals2[1]));
        $this->assertNotEquals(spl_object_id($gru2vals[2]),spl_object_id($gru2vals2[2]));
        $this->assertNotEquals(spl_object_id($densevals[0]),spl_object_id($densevals2[0]));
        $this->assertNotEquals(spl_object_id($densevals[1]),spl_object_id($densevals2[1]));

        $this->assertEquals(spl_object_id($embvals1[0]),spl_object_id($embvals2[0]));
        $this->assertEquals(spl_object_id($gruvals1[0]),spl_object_id($gruvals2[0]));
        $this->assertEquals(spl_object_id($gruvals1[1]),spl_object_id($gruvals2[1]));
        $this->assertEquals(spl_object_id($gruvals1[2]),spl_object_id($gruvals2[2]));
        $this->assertEquals(spl_object_id($gru2vals1[0]),spl_object_id($gru2vals2[0]));
        $this->assertEquals(spl_object_id($gru2vals1[1]),spl_object_id($gru2vals2[1]));
        $this->assertEquals(spl_object_id($gru2vals1[2]),spl_object_id($gru2vals2[2]));
        $this->assertEquals(spl_object_id($densevals1[0]),spl_object_id($densevals2[0]));
        $this->assertEquals(spl_object_id($densevals1[1]),spl_object_id($densevals2[1]));

        $this->assertLessThan(1e-7,$mo->la()->sum($mo->la()->square($mo->op($y,'-',$y2))));
        //$this->assertEquals($t->toArray(),$mo->argMax($y,axis:1)->toArray());
    }

    public function testSaveAndLoadModelPortable()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,input_shape:[2]),
            $nn->layers()->BatchNormalization(),
            $nn->layers()->Activation('sigmoid'),
            $nn->layers()->Dense($units=2, activation:'softmax'),
        ]);
        $model->compile();
        $x = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $t = $mo->array([0, 0, 0, 1, 1, 1],dtype:NDArray::int32);
        $history = $model->fit($x,$t, epochs:100, verbose:0);
        $y = $model->predict($x);
        $evals = $model->evaluate($x,$t);

        $model->save($this->filename,$portable=true);

        // load model
        $model = $nn->models()->loadModel($this->filename);

        $z = $model->predict($x);
        if($this->plot) {
            [$fig,$ax] = $plt->subplots(2);
            $diff = $mo->f('abs',$mo->select(
                $mo->op($y,'-',$z),
                $mo->arange($t->size(),dtype:NDArray::int32),
                $mo->zeros([$t->size()],dtype:NDArray::int32)
            ));
            $ax[0]->bar($mo->arange($diff->size(),dtype:NDArray::int32),$diff,null,null,'difference');
            $ax[0]->legend();
            $ax[1]->plot($mo->array($history['loss']),null,null,'loss');
            $ax[1]->plot($mo->array($history['accuracy']),null,null,'accuracy');
            $ax[1]->legend();
            $plt->title('save portable mode');
            $plt->show();
        }

        $evals2 = $model->evaluate($x,$t);
        $this->assertLessThan(0.5,abs($evals['loss']-$evals2['loss']));
        $this->assertLessThan(0.5,abs($evals['accuracy']-$evals2['accuracy']));
        //$this->assertEquals($t->toArray(),$mo->argMax($y,axis:1)->toArray());
    }
}
