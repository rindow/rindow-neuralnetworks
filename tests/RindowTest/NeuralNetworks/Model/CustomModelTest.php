<?php
namespace RindowTest\NeuralNetworks\Model\CustomModelTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Model\ModelLoader;
use Rindow\NeuralNetworks\Model\AbstractModel;
use Rindow\NeuralNetworks\Layer\AbstractLayer;
use Rindow\NeuralNetworks\Layer\Flatten;
use Rindow\NeuralNetworks\Layer\Dense;
use Interop\Polite\Math\Matrix\NDArray;
use PDO;

class TestModel extends AbstractModel
{
    protected $flatten;
    protected $custom;
    protected $fc;

    public function __construct($backend,$builder)
    {
        parent::__construct(
            $backend,
            $builder,
            $builder->utils()->HDA());
        $this->flatten = $builder->layers()->Flatten(input_shape:[5]);
        $this->custom = new TestSubModel($backend,$builder);
        $this->fc = $builder->layers()->Dense(
            10,
            activation:'softmax'
        );
        //$this->setLastLayer($this->fc);
    }

    //protected function buildLayers(array $options=null) : void
    //{
    //    $shape = $this->registerLayer($this->flatten);
    //    $shape = $this->registerLayer($this->custom,$shape);
    //    $shape = $this->registerLayer($this->fc,$shape);
    //}

    protected function call($inputs, $training=null, $trues=null)
    {
        $training = $training ?? false;
        $flat = $this->flatten->forward($inputs,$training);
        $customout = $this->custom->forward($flat,$training);
        $outputs = $this->fc->forward($customout,$training);
        return $outputs;
    }

    protected function differentiate(NDArray $dout) : NDArray
    {
        $dout = $this->fc->backward($dout);
        $dout = $this->custom->backward($dout);
        $dInputs = $this->flatten->backward($dout);
        return $dInputs;
    }
}

class TestSubModel extends AbstractModel
{
    protected $fc;
    public function __construct($backend,$builder)
    {
        $this->backend = $backend;
        $this->fc = $builder->layers()->Dense(5);
    }

    //public function build(array $inputShape=null, array $options=null) : array
    //{
    //    $inputShape=$this->normalizeInputShape($inputShape);
    //    $shape = $this->registerLayer($this->fc,$inputShape);
    //    $this->outputShape = $shape;
    //    return $this->outputShape;
    //}

    protected function call($inputs, $training)
    {
        $out = $this->fc->forward($inputs,$training);
        return $out;
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $din = $this->fc->backward($dOutputs);
        return $din;
    }
}

class TestRNNModel extends AbstractModel
{
    protected $embed0;
    protected $rnn0;
    protected $embed1;
    protected $rnn1;
    protected $attention;
    protected $concat;
    protected $dense;
    protected $activation;

    public function __construct($backend,$builder)
    {
        parent::__construct(
            $backend,
            $builder,
            $builder->utils()->HDA());
        $this->embed0 = $builder->layers->Embedding(
            $inputDim=5, $outputDim=4,
            input_length:3);
        $this->rnn0 = $builder->layers->GRU($units=32,
            return_state:true, return_sequences:true,
            recurrent_initializer:'glorot_uniform'
        );
        $this->embed1 = $builder->layers->Embedding(
            $inputDim=5, $outputDim=4,
            input_length:3);
        $this->rnn1 = $builder->layers->GRU($units=32,
            return_state:true,return_sequences:true,
            recurrent_initializer:'glorot_uniform'
        );
        $this->attention = $builder->layers->Attention();
        $this->concat = $builder->layers->Concatenate();
        $this->dense = $builder->layers->Dense($vocabSize=8);
        $this->activation = $builder->layers->Activation('softmax');
    }

    protected function call($inputs, $training=null, $trues=null)
    {
        // encoder
        $x = $this->embed0->forward($inputs,$training);
        [$encOutputs,$encStates] = $this->rnn0->forward($x,$training);
        // decoder
        $targets = $this->embed1->forward($trues,$training);
        [$rnnSequence,$states] = $this->rnn1->forward($targets,$training,$encStates);
        $contextVector = $this->attention->forward([$rnnSequence,$encOutputs],$training);
        $outputs = $this->concat->forward([$contextVector, $rnnSequence],$training);
        $outputs = $this->dense->forward($outputs,$training);
        $outputs = $this->activation->forward($outputs,$training);
        return $outputs;
    }

    public function shiftLeftSentence(
        NDArray $sentence
        ) : NDArray
    {
        $K = $this->backend;
        $shape = $sentence->shape();
        $batchs = $shape[0];
        $zeroPad = $K->zeros([$batchs,1],$sentence->dtype());
        $seq = $K->slice($sentence,[0,1],[-1,-1]);
        $result = $K->concat([$seq,$zeroPad],$axis=1);
        return $result;
    }

    protected function loss(NDArray $trues,NDArray $preds) : float
    {
        $trues = $this->shiftLeftSentence($trues);
        return parent::loss($trues,$preds);
    }

    protected function accuracy(NDArray $trues,NDArray $preds) : float
    {
        $trues = $this->shiftLeftSentence($trues);
        return parent::accuracy($trues,$preds);
    }
}

class TestMultiInputModel extends AbstractModel
{
    protected $inp1;
    protected $inp2;
    protected $concat;
    protected $fc;
    public function __construct($backend,$builder)
    {
        parent::__construct(
            $backend,
            $builder,
            $builder->utils()->HDA());
        $this->inp1 = $builder->layers()->Flatten(input_shape:[2]);
        $this->inp2 = $builder->layers()->Flatten(input_shape:[2]);
        $this->concat = $builder->layers()->Concatenate();
        $this->fc = $builder->layers()->Dense(5,activation:'softmax');
    }

    protected function call($inp1,$inp2,$training)
    {
        $inp1 = $this->inp1->forward($inp1,$training);
        $inp2 = $this->inp2->forward($inp2,$training);
        $x = $this->concat->forward([$inp1,$inp2],$training);
        $out = $this->fc->forward($x,$training);
        return $out;
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $din = $this->fc->backward($dOutputs);
        return $din;
    }
}

class Test extends TestCase
{
    protected $filename;
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

    public function initSaveModel() : void
    {
        $this->filename = __DIR__.'/../../../tmp/savedcustommodel.hda.sqlite3';
        $pdo = new PDO('sqlite:'.$this->filename);
        $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
        $sql = "DROP TABLE IF EXISTS hda";
        $stat = $pdo->exec($sql);
        unset($stat);
        unset($pdo);
    }

    public function testComplieAndFitNormal()
    {
        Flatten::$nameNumbering = 0;
        Dense::$nameNumbering = 0;
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $model = new TestModel($K,$nn);

        $train = $mo->random()->randn([10,5]);
        $label = $mo->arange(10);
        $val_train = $mo->random()->randn([10,5]);
        $val_label = $mo->arange(10);

        $model->compile();
        $layers = $model->layers();
        $this->assertCount(3,$layers);
        $outputs = $nn->with($tape=$g->GradientTape(),function () use ($K,$model,$train) {
            $train = $K->array($train);
            $outputs = $model->forward($train);
            return $outputs;
        });
        $layers = [];
        $f = $outputs->creator();
        array_unshift($layers,$f);
        $f = $f->inputs()[0]->creator();
        array_unshift($layers,$f);
        $f = $f->inputs()[0]->creator();
        array_unshift($layers,$f);
        $this->assertEquals('flatten',$layers[0]->func()->getName());
        $this->assertEquals('dense',$layers[1]->func()->getName());
        $this->assertEquals('dense_1',$layers[2]->func()->getName());
        $this->assertEquals(0,$layers[0]->generation());
        $this->assertEquals(1,$layers[1]->generation());
        $this->assertEquals(2,$layers[2]->generation());
        //$model->summary();
        $history = $model->fit(
            $train,$label,
            epochs:5,batch_size:2,validation_data:[$val_train,$val_label],verbose:0
        );
        $this->assertTrue(true);
    }

    public function testComplieAndFitRNNLayers()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);

        $model = new TestRNNModel($K,$nn);
        $inputs = $mo->array(
            [[1, 3, 3], [1, 4, 3], [2, 4, 4], [3, 1, 4], [4, 1, 4], [4, 2, 2]],
            NDArray::int32
        );
        $targets = $mo->array(
            [[3, 1, 1], [4, 1, 4], [4, 2, 2], [1, 3, 2], [1, 4, 4], [2, 4, 3]],
            NDArray::int32
        );

        $model->compile(
            loss:'sparse_categorical_crossentropy',
            optimizer:'adam',
        );
        $layers = $model->layers();
        //$model->summary();
        $this->assertCount(8,$layers);

        $history = $model->fit(
            $inputs, $targets,
            batch_size:2,epochs:10,shuffle:true,verbose:0);

        $this->assertTrue(true);
    }

    public function testSaveAndLoadNormal()
    {
        $this->initSaveModel();

        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $backend = $this->newBackend($nn);

        $model = new TestModel($backend,$nn);

        $train = $mo->random()->randn([10,5]);
        $label = $mo->arange(10);
        $val_train = $mo->random()->randn([10,5]);
        $val_label = $mo->arange(10);

        $model->compile();
        $history = $model->fit(
            $train,$label,
            epochs:5,batch_size:2,validation_data:[$val_train,$val_label],verbose:0
        );
        $weightsOriginal = ['layers'=>[],'optimizer'=>[]];
        foreach ($model->trainableVariables() as $key => $var) {
            $value = $var->value();
            $weightsOriginal['layers'][$key] = $backend->ndarray($value);
        };
        foreach ($model->optimizer()->getWeights() as $key => $value) {
            $weightsOriginal['optimizer'][$key] = $backend->ndarray($value);
        };

        $model->saveWeightsToFile($this->filename);

        // load model
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $backend = $this->newBackend($nn);

        $model = new TestModel($backend,$nn);
        $train = $mo->random()->randn([10,5]);
        $label = $mo->arange(10);
        $val_train = $mo->random()->randn([10,5]);
        $val_label = $mo->arange(10);

        $model->compile();
        //$model->summary();
        $model->loadWeightsFromFile($this->filename);

        foreach ($model->trainableVariables() as $key => $var) {
            $value = $var->value();
            $this->assertEquals($value->toArray(),$weightsOriginal['layers'][$key]->toArray());
        }
        foreach ($model->optimizer()->getWeights() as $key => $value) {
            $this->assertEquals($value->toArray(),$weightsOriginal['optimizer'][$key]->toArray());
        }
    }

    public function testSaveAndLoadRNNLayers()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $backend = $this->newBackend($nn);

        $model = new TestRNNModel($backend,$nn);
        $inputs = $mo->array(
            [[1, 3, 3], [1, 4, 3], [2, 4, 4], [3, 1, 4], [4, 1, 4], [4, 2, 2]],
            NDArray::int32
        );
        $targets = $mo->array(
            [[3, 1, 1], [4, 1, 4], [4, 2, 2], [1, 3, 2], [1, 4, 4], [2, 4, 3]],
            NDArray::int32
        );

        $model->compile(
            loss: 'sparse_categorical_crossentropy',
            optimizer: 'adam',
        );
        $layers = $model->layers();
        //$model->summary();
        $this->assertCount(8,$layers);

        $history = $model->fit(
            $inputs, $targets,
            batch_size:2, epochs:10, shuffle: true, verbose: 0
        );

        $weightsOriginal = ['layers'=>[],'optimizer'=>[]];
        foreach ($model->trainableVariables() as $key => $var) {
            $value = $var->value();
            $weightsOriginal['layers'][$key] = $backend->ndarray($value);
        };
        foreach ($model->optimizer()->getWeights() as $key => $value) {
            $weightsOriginal['optimizer'][$key] = $backend->ndarray($value);
        };
        $model->saveWeightsToFile($this->filename);

        // load model
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $backend = $this->newBackend($nn);

        $model = new TestRNNModel($backend,$nn);
        $inputs = $mo->array(
            [[1, 3, 3], [1, 4, 3], [2, 4, 4], [3, 1, 4], [4, 1, 4], [4, 2, 2]],
            NDArray::int32
        );
        $targets = $mo->array(
            [[3, 1, 1], [4, 1, 4], [4, 2, 2], [1, 3, 2], [1, 4, 4], [2, 4, 3]],
            NDArray::int32
        );

        $model->compile(
            loss: 'sparse_categorical_crossentropy',
            optimizer: 'adam',
        );
        //$model->summary();

        $model->loadWeightsFromFile($this->filename);

        foreach ($model->trainableVariables() as $key => $var) {
            $value = $var->value();
            $this->assertEquals($value->toArray(),$weightsOriginal['layers'][$key]->toArray());
        }
        foreach ($model->optimizer()->getWeights() as $key => $value) {
            $this->assertEquals($value->toArray(),$weightsOriginal['optimizer'][$key]->toArray());
        }
    }

    public function testCloneNest()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $backend = $this->newBackend($nn);

        $origModel = new TestModel($backend,$nn);

        // before build
        $model = clone $origModel;
        $origModel->compile();
        $model->compile();

        $origParams = $origModel->trainableVariables();
        $params = $model->trainableVariables();
        $this->assertCount(2+2,$params);
        foreach (array_map(null,$origParams,$params) as [$orig,$dest]) {
            $this->assertNotEquals(spl_object_id($orig),spl_object_id($dest));
        }

        // after build
        $train = $mo->random()->randn([10,5]);
        $label = $mo->arange(10);
        $val_train = $mo->random()->randn([10,5]);
        $val_label = $mo->arange(10);
        $history = $origModel->fit(
            $train,$label,
            epochs:5,batch_size:2,validation_data:[$val_train,$val_label],verbose:0
        );
        $model = clone $origModel;

        $origParams2 = $origModel->trainableVariables();
        foreach (array_map(null,$origParams,$origParams2) as [$before,$after]) {
            $this->assertEquals(spl_object_id($before),spl_object_id($after));
        }
        $params = $model->trainableVariables();
        $this->assertCount(2+2,$params);
        foreach (array_map(null,$origParams,$params) as [$orig,$dest]) {
            $this->assertNotEquals(spl_object_id($orig),spl_object_id($dest));
            $this->assertNotEquals(spl_object_id($orig->value()),spl_object_id($dest->value()));
        }

        //$origParams = $origModel->grads();
        //$params = $model->grads();
        //$this->assertCount(2+2,$params);
        //foreach (array_map(null,$origParams,$params) as $data) {
        //    [$orig,$dest] = $data;
        //    $this->assertNotEquals(spl_object_id($orig),spl_object_id($dest));
        //}
    }

    public function testCloneRNN()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $backend = $this->newBackend($nn);

        $origModel = new TestRNNModel($backend,$nn);

        $model = clone $origModel;
        $origModel->compile(
            loss: 'sparse_categorical_crossentropy',
            optimizer: 'adam',
        );
        $model->compile(
            loss: 'sparse_categorical_crossentropy',
            optimizer: 'adam',
        );

        // before build
        $origParams = $origModel->trainableVariables();
        $params = $model->trainableVariables();
        $this->assertCount(1+3+1+3+2,$params);
        foreach (array_map(null,$origParams,$params) as [$orig,$dest]) {
            $this->assertNotEquals(spl_object_id($orig),spl_object_id($dest));
        }

        // after build
        $inputs = $mo->array(
            [[1, 3, 3], [1, 4, 3], [2, 4, 4], [3, 1, 4], [4, 1, 4], [4, 2, 2]],
            NDArray::int32
        );
        $targets = $mo->array(
            [[3, 1, 1], [4, 1, 4], [4, 2, 2], [1, 3, 2], [1, 4, 4], [2, 4, 3]],
            NDArray::int32
        );
        $history = $origModel->fit(
            $inputs, $targets,
            batch_size: 2, epochs: 10, shuffle: true, verbose: 0);


        $model = clone $origModel;

        $origParams2 = $origModel->trainableVariables();
        foreach (array_map(null,$origParams,$origParams2) as [$before,$after]) {
            $this->assertEquals(spl_object_id($before),spl_object_id($after));
        }
        $params = $model->trainableVariables();
        $this->assertCount(1+3+1+3+2,$params);
        foreach (array_map(null,$origParams,$params) as [$orig,$dest]) {
            $this->assertNotEquals(spl_object_id($orig),spl_object_id($dest));
            $this->assertNotEquals(spl_object_id($orig->value()),spl_object_id($dest->value()));
        }

        //$origParams = $origModel->grads();
        //$params = $model->grads();
        //$this->assertCount(1+3+1+3+2,$params);
        //foreach (array_map(null,$origParams,$params) as $data) {
        //    [$orig,$dest] = $data;
        //    $this->assertNotEquals(spl_object_id($orig),spl_object_id($dest));
        //}
    }

    public function testMultiInput()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $backend = $this->newBackend($nn);

        $model = new TestMultiInputModel($backend,$nn);
        $model->compile(numInputs:2);
        //$model->summary();
        $a = $mo->zeros([3,2]);
        $b = $mo->ones([3,2]);
        $t = $mo->zeros([3],NDArray::int32);
        $model->fit([$a,$b],$t, epochs:1,verbose:0);
        $out = $model->predict([$a,$b]);
        $this->assertEquals([3,5],$out->shape());
        $model->evaluate([$a,$b],$t);
    }

    public function testSummary()
    {
        Flatten::$nameNumbering = 0;
        Dense::$nameNumbering = 0;
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $nn->backend();
        $g = $nn->gradient();

        $model = new TestModel($K,$nn);

        
        $model->build([1,5]);
        ob_start();
        $model->summary();
        $dump = ob_get_clean();
        $display = 
        'Layer(type)                  Output Shape               Param #   '."\n".
        '=================================================================='."\n".
        'flatten(Flatten)             (5)                        0         '."\n".
        'dense(Dense)                 (5)                        30        '."\n".
        'dense_1(Dense)               (10)                       60        '."\n".
        '=================================================================='."\n".
        'Total params: 90'."\n";

        $this->assertEquals($display,$dump);
    }
}
