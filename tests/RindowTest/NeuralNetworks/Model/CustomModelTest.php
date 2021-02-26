<?php
namespace RindowTest\NeuralNetworks\Model\CustomModelTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Model\ModelLoader;
use Rindow\NeuralNetworks\Model\AbstractModel;
use Rindow\NeuralNetworks\Layer\AbstractLayer;
use Interop\Polite\Math\Matrix\NDArray;
use PDO;

class TestModel extends AbstractModel
{
    protected $flat;
    protected $custom;
    protected $fc;

    public function __construct($backend,$builder)
    {
        parent::__construct(
            $backend,
            $builder,
            $builder->utils()->HDA());
        $this->flatten = $builder->layers()->Flatten(['input_shape'=>[5]]);
        $this->custom = new TestLayer($backend,$builder);
        $this->fc = $builder->layers()->Dense(
            10,
            ['activation'=>'softmax']
        );
        $this->setLastLayer($this->fc);
    }

    protected function buildLayers(array $options=null) : void
    {
        $shape = $this->registerLayer($this->flatten);
        $shape = $this->registerLayer($this->custom,$shape);
        $shape = $this->registerLayer($this->fc,$shape);
    }

    protected function forwardStep(NDArray $inputs, NDArray $trues=null, bool $training=null) : NDArray
    {
        $flat = $this->flatten->forward($inputs,$training);
        $customout = $this->custom->forward($flat,$training);
        $outputs = $this->fc->forward($customout,$training);
        return $outputs;
    }

    protected function backwardStep(NDArray $dout) : NDArray
    {
        $dout = $this->fc->backward($dout);
        $dout = $this->custom->backward($dout);
        $dInputs = $this->flatten->backward($dout);
        return $dInputs;
    }
}

class TestLayer extends AbstractLayer
{
    public function __construct($backend,$builder)
    {
        $this->backend = $backend;
        $this->fc = $builder->layers()->Dense(5);
    }

    public function build(array $inputShape=null, array $options=null) : array
    {
        $inputShape=$this->normalizeInputShape($inputShape);
        $shape = $this->registerLayer($this->fc,$inputShape);
        $this->outputShape = $shape;
        return $this->outputShape;
    }

    protected function call(NDArray $inputs,bool $training) : NDArray
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

class Test extends TestCase
{
    public function newBackend($mo)
    {
        $builder = new NeuralNetworks($mo);
        return $builder->backend();
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

    public function testComplieAndFit()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $model = new TestModel($backend,$nn);

        $train = $mo->random()->randn([10,5]);
        $label = $mo->arange(10);
        $val_train = $mo->random()->randn([10,5]);
        $val_label = $mo->arange(10);

        $model->compile();
        $history = $model->fit(
            $train,$label,
            ['epochs'=>5,'batch_size'=>2,'validation_data'=>[$val_train,$val_label],'verbose'=>0]
        );
        $this->assertTrue(true);
    }

    public function testSaveAndLoad()
    {
        $this->initSaveModel();

        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $model = new TestModel($backend,$nn);

        $train = $mo->random()->randn([10,5]);
        $label = $mo->arange(10);
        $val_train = $mo->random()->randn([10,5]);
        $val_label = $mo->arange(10);

        $model->compile();
        $history = $model->fit(
            $train,$label,
            ['epochs'=>5,'batch_size'=>2,'validation_data'=>[$val_train,$val_label],'verbose'=>0]
        );
        $weightsOriginal = ['layers'=>[],'optimizer'=>[]];
        foreach ($model->weights() as $key => $value) {
            $weightsOriginal['layers'][$key] = $backend->ndarray($value);
        };
        foreach ($model->optimizer()->getWeights() as $key => $value) {
            $weightsOriginal['optimizer'][$key] = $backend->ndarray($value);
        };

        $model->saveWeightsToFile($this->filename);

        // load model
        $model = new TestModel($backend,$nn);
        $train = $mo->random()->randn([10,5]);
        $label = $mo->arange(10);
        $val_train = $mo->random()->randn([10,5]);
        $val_label = $mo->arange(10);

        $model->compile();
        $history = $model->fit(
            $train,$label,
            ['epochs'=>5,'batch_size'=>2,'validation_data'=>[$val_train,$val_label],'verbose'=>0]
        );
        $model->loadWeightsFromFile($this->filename);
        foreach ($model->weights() as $key => $value) {
            $this->assertEquals($value->toArray(),$weightsOriginal['layers'][$key]->toArray());
        }
        foreach ($model->optimizer()->getWeights() as $key => $value) {
            $this->assertEquals($value->toArray(),$weightsOriginal['optimizer'][$key]->toArray());
        }
    }
}
