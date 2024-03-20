<?php
namespace RindowTest\NeuralNetworks\Gradient\Model\ModelTest;

use InvalidArgumentException;
use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Model\AbstractModel;
use Rindow\Math\Plot\Plot;

use Rindow\NeuralNetworks\Gradient\Core\GradientTape;

class TestModel1 extends AbstractModel
{
    protected $dense;
    public function __construct(
        $builder
        )
    {
        parent::__construct($builder);
        $this->dense = $builder->layers->Dense($units=5,input_shape:[1]);
    }

    protected function call($inputs)
    {
        $dense = $this->dense;
        $outputs = $dense($inputs);
        return $outputs;
    }
}

class TestModel2 extends AbstractModel
{
    protected $dense1;
    protected $dense2;
    public function __construct(
        $builder
        )
    {
        parent::__construct($builder);
        $this->dense1 = $builder->layers->Dense($units=128,
                input_shape:[2], activation:'sigmoid'
            );
        $this->dense2 = $builder->layers->Dense($units=2);
    }

    protected function call($inputs)
    {
        $dense1 = $this->dense1;
        $dense2 = $this->dense2;
        $x = $dense1($inputs);
        $outputs = $dense2($x);
        return $outputs;
    }
}

class Test3Mini1 extends AbstractModel
{
    protected $dense1;
    public function __construct(
        $builder
        )
    {
        parent::__construct($builder);
        $this->dense1 = $builder->layers->Dense($units=128,
                input_shape:[2], activation:'sigmoid'
            );
    }
    protected function call($inputs)
    {
        $dense1 = $this->dense1;
        $outputs = $dense1($inputs);
        return $outputs;
    }
}

class Test3Mini2 extends AbstractModel
{
    protected $dense2;
    public function __construct(
        $builder
        )
    {
        parent::__construct($builder);
        $this->dense2 = $builder->layers->Dense($units=2);
    }
    protected function call($inputs)
    {
        $dense2 = $this->dense2;
        $outputs = $dense2($inputs);
        return $outputs;
    }
}

class Test3Main extends AbstractModel
{
    protected $model1;
    protected $model2;
    public function __construct(
        $builder
        )
    {
        parent::__construct($builder);
        $this->model1 = new Test3Mini1($builder);
        $this->model2 = new Test3Mini2($builder);
    }

    protected function call($inputs)
    {
        $model1 = $this->model1;
        $model2 = $this->model2;
        $x = $model1($inputs);
        $outputs = $model2($x);
        return $outputs;
    }
}

class TestRNNEncoder extends AbstractModel
{
    protected $embed;
    protected $rnn;
    public function __construct(
        $builder
        )
    {
        parent::__construct($builder);
        $this->embed = $builder->layers->Embedding($inputDim=5, $outputDim=4,
                input_length:3
            );
        $this->rnn = $builder->layers->LSTM($units=32,
                return_state:true,return_sequences:true,
                recurrent_initializer:'glorot_uniform'
            );
    }

    protected function call($inputs,$initial_states=null)
    {
        $embed = $this->embed;
        $rnn = $this->rnn;

        $x = $embed($inputs);
        [$outputs,$states] = $rnn($x,initialStates:$initial_states);
        return [$outputs, $states];
    }
}

class TestRNNDecoder extends AbstractModel
{
    protected $embed;
    protected $rnn;
    protected $attention;
    protected $concat;
    protected $dense;
    protected $attentionScores;
    public function __construct(
        $builder
        )
    {
        parent::__construct($builder);
        $this->embed = $builder->layers->Embedding($inputDim=5, $outputDim=4,
                input_length:3
            );
        $this->rnn = $builder->layers->LSTM($units=32,
                return_state:true,return_sequences:true,
                recurrent_initializer:'glorot_uniform'
            );
        $this->attention = $builder->layers->Attention();
        $this->concat = $builder->layers->Concatenate();
        $this->dense = $builder->layers->Dense($units=8);
    }

    protected function call($inputs,$encOutputs,$encStates,$returnAttentionScores=null)
    {
        $embed = $this->embed;
        $rnn = $this->rnn;
        $attention = $this->attention;
        $concat = $this->concat;
        $dense = $this->dense;

        $x = $embed($inputs);
        [$rnnSequence,$states] = $rnn($x,initialStates:$encStates);
        $contextVector = $attention([$rnnSequence,$encOutputs],
                                        returnAttentionScores:$returnAttentionScores);
        if(is_array($contextVector)) {
            [$contextVector,$attentionScores] = $contextVector;
            $this->attentionScores = $attentionScores;
        }
        $outputs = $concat([$contextVector, $rnnSequence]);
        $outputs = $dense($outputs);
        return [$outputs,$states];
    }

    public function getAttentionScores()
    {
        return $this->attentionScores;
    }
}

class TestRNNMain extends AbstractModel
{
    protected $mo;
    protected $encoder;
    protected $decoder;
    protected $out;
    public function __construct(
        $mo,
        $builder
        )
    {
        parent::__construct($builder);
        $this->mo = $mo;
        $this->encoder = new TestRNNEncoder($builder);
        $this->decoder = new TestRNNDecoder($builder);
        $this->out = $builder->layers->Activation('softmax');
    }

    protected function call($inputs,$trues=null)
    {
        $encoder = $this->encoder;
        $decoder = $this->decoder;
        $out = $this->out;
        //$trues =$this->builder->gradient->Variable($trues);

        [$encOutputs,$states] = $encoder($inputs);
        [$outputs,$dmyStatus] = $decoder($trues,$encOutputs,$states);
        $outputs = $out($outputs);
        return $outputs;
    }

    public function shiftLeftSentence(
        NDArray $sentence
        ) : NDArray
    {
        $K = $this->backend();
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

    public function predict(
        $inputs, 
        array|object $callbacks=null, 
        ...$options
    ) : NDArray
    {
        $K = $this->backend();
        $g = $this->builder->gradient();
        $encoder = $this->encoder;
        $decoder = $this->decoder;
        $attentionPlot = $options['attention_plot'];

        if($inputs->ndim()!=2) {
            throw new InvalidArgumentException('inputs shape must be 2D.');
        }
        $batchs = $inputs->shape()[0];
        if($batchs!=1) {
            throw new InvalidArgumentException('num of batch must be one.');
        }
        $inputs = $g->Variable($K->array($inputs));
        $status = [
            $g->Variable($K->zeros([$batchs, 32])),
            $g->Variable($K->zeros([$batchs, 32]))
        ];
        [$encOutputs, $status] = $encoder($inputs, $status);

        $decInputs = $g->Variable($K->array([[0]],$inputs->dtype()));

        $result = [];
        $this->setShapeInspection(false);
        for($t=0;$t<3;$t++) {
            [$predictions, $status] = $decoder(
                $decInputs, $encOutputs, $status,
                returnAttentionScores:true);

            # storing the attention weights to plot later on
            $scores = $decoder->getAttentionScores();
            $this->mo->la()->copy($K->ndarray($scores->reshape([3])),$attentionPlot[$t]);

            $predictions = $predictions->value();
            $predictedId = $K->scalar($K->argmax($predictions[0][0]));

            $result[] = $predictedId;

            if(0 == $predictedId)
                break;

            # the predicted ID is fed back into the model
            $decInputs = $g->Variable($K->array([[$predictedId]],$inputs->dtype()));
        }
        $this->setShapeInspection(true);
        $result = $K->array([$result],NDArray::int32);
        #return result, sentence, attention_plot
        return $K->ndarray($result);
    }
}

class TestVariableMini1 extends AbstractModel
{
    protected $linearWeight;
    protected $linearBias;
    protected $activation;
    public function __construct(
        $builder
        )
    {
        parent::__construct($builder);
        $backend = $this->backend();
        $g = $builder->gradient();
        $inputDim = 2;
        $units = 128;
        $kernelInitializer = $backend->getInitializer('glorot_uniform');
        $weights = $kernelInitializer([$inputDim,$units],[$inputDim,$units]);
        $this->linearWeight = $g->Variable($weights,name:'W1');
        $this->linearBias = $g->Variable($backend->zeros([$units]),name:'B1');
        $this->activation = $builder->layers->Activation('sigmoid', input_shape:[128]);
    }

    protected function call($inputs)
    {
        $K = $this->backend();
        $g = $this->builder->gradient();
        $activation = $this->activation;
        $outputs = $g->matmul($inputs,$this->linearWeight);
        $outputs = $g->add($outputs,$this->linearBias);
        $outputs = $activation($outputs,true);
        return $outputs;
    }
}

class TestVariableMini2 extends AbstractModel
{
    protected $linearWeight;
    protected $linearBias;
    public function __construct(
        $builder
        )
    {
        parent::__construct($builder);
        $backend = $this->backend();
        $g = $builder->gradient();
        $inputDim = 128;
        $units = 2;
        $kernelInitializer = $backend->getInitializer('glorot_uniform');
        $weights = $kernelInitializer([$inputDim,$units],[$inputDim,$units]);
        $this->linearWeight = $g->Variable($weights,name:'W2');
        $this->linearBias = $g->Variable($backend->zeros([$units]),name:'B2');
    }

    protected function call($inputs)
    {
        $K = $this->backend;
        $g = $this->builder->gradient();
        $outputs = $g->matmul($inputs,$this->linearWeight);
        $outputs = $g->add($outputs,$this->linearBias);
        return $outputs;
    }
}

class TestVariableMain extends AbstractModel
{
    protected $model1;
    protected $model2;
    public function __construct(
        $builder
        )
    {
        parent::__construct($builder);
        $this->model1 = new TestVariableMini1($builder);
        $this->model2 = new TestVariableMini2($builder);
    }

    protected function call($inputs)
    {
        $model1 = $this->model1;
        $model2 = $this->model2;
        $x = $model1($inputs);
        $outputs = $model2($x);
        return $outputs;
    }
}

class TestGraphMode extends AbstractModel
{
    protected $log = [];
    protected $in;
    protected $fc;

    public function __construct(
        $builder
        )
    {
        parent::__construct($builder);
        $nn = $builder;
        $this->in = $nn->layers->Input(shape:[2]);
        $this->fc = $nn->layers->Dense(3);
    }

    protected function call($inputs)
    {
        $this->log('call');
        $in = $this->in;
        $fc = $this->fc;
        $x = $in($inputs);
        $outputs = $fc($x);
        return $outputs;
    }

    public function log($message)
    {
        $this->log[] = $message;
    }

    public function getLog()
    {
        return $this->log;
    }
}

class ModelTest extends TestCase
{
    protected $plot = true;

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

    public function getPlotConfig()
    {
        return [
            'renderer.skipCleaning' => true,
            'renderer.skipRunViewer' => getenv('PLOT_RENDERER_SKIP') ? true : false,
        ];
    }

    public function testGradient()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $x = $g->Variable($K->array([[3.0], [4.0]]));
        $model = new TestModel1($nn);

        $outputs = $nn->with($tape=$g->GradientTape(),
            function() use ($model,$x) {
                $outputs = $model($x,true);
                return $outputs;
            }
        );
        $gradients = $tape->gradient($outputs, $model->trainableVariables());
        //$optimizer->update($model->params(),$gradients);
        $this->assertCount(2,$gradients);
        $this->assertEquals([1,5],$gradients[0]->shape());
        $this->assertEquals([5],$gradients[1]->shape());
        $this->assertEquals("[[7,7,7,7,7]]",$mo->toString($gradients[0]));
        $this->assertEquals("[2,2,2,2,2]",$mo->toString($gradients[1]));
    }

    public function testManualUpdate()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = new TestModel2($nn);
        $lossfunc = $nn->losses->SparseCategoricalCrossentropy(from_logits:true);
        $optimizer = $nn->optimizers->Adam();
        $train_inputs = $K->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $train_tests = $K->array([0, 0, 0, 1, 1, 1],dtype:NDArray::int32);
        $dataset = $nn->data->NDArrayDataset($train_inputs,
            tests:$train_tests,
            batch_size:64,
            shuffle:false,
        );
        $history = [];
        for($epoch=0;$epoch<100;$epoch++) {
            $totalLoss = 0;
            $paramsum = 0;
            foreach($dataset as $batchIndex => [$inputs,$trues]) {
                $inputs = $K->array($inputs);
                $trues = $K->array($trues);
                $x = $g->Variable($inputs);
                $t = $g->Variable($trues);
                [$loss,$predicts] = $nn->with($tape=$g->GradientTape(),
                    function() use ($epoch,$K,$model,$lossfunc,$x,$t,$trues) {
                        $predicts = $model($x,true,$t);
                        return [$lossfunc($trues,$predicts),$predicts];
                    }
                );
                $params = $model->trainableVariables();
                $gradients = $tape->gradient($loss, $params);

                $optimizer->update($params,$gradients);
                $totalLoss += $K->scalar($loss->value());
            }
            $history[] = $totalLoss;
        }
        if($this->plot) {
            $plt->plot($mo->array($history),null,null,'loss');
            $plt->legend();
            $plt->title('dynamic mode gradient');
            $plt->show();
        }
        $this->assertTrue(true);
        //$optimizer->update($model->params(),$gradients);
    }

    public function testFitDefault()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = new TestModel2($nn);
        $lossfunc = $nn->losses->SparseCategoricalCrossentropy(from_logits:true);
        $train_inputs = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $train_tests = $mo->array([0, 0, 0, 1, 1, 1],dtype:NDArray::int32);
        $model->compile(
            loss: $lossfunc,
            optimizer: 'adam',
        );

        $history = $model->fit(
            $train_inputs, $train_tests,
            batch_size:2, epochs: 100, shuffle:true, verbose:0);

        if($this->plot) {
            $plt->plot($mo->array($history['loss']),null,null,'loss');
            $plt->plot($mo->array($history['accuracy']),null,null,'accuracy');
            $plt->legend();
            $plt->title('fit at dynamic mode');
            $plt->show();
        }
        $this->assertTrue(true);
        //$optimizer->update($model->params(),$gradients);
    }

    public function testEvaluate()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = new TestModel2($nn);

        $lossfunc = $nn->losses->SparseCategoricalCrossentropy(from_logits:true);
        $model->compile(
            loss:$lossfunc,
            optimizer:'adam',
        );

        // training greater or less
        $x = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $t = $mo->array([0, 0, 0, 1, 1, 1],dtype:NDArray::int32);
        $history = $model->fit($x,$t,epochs:100, verbose:0);

        $evals = $model->evaluate($x,$t);
        $this->assertLessThan(1.0,$evals['loss']);
        $this->assertEquals(1.0,$evals['accuracy']);
    }

    public function testFitWithEvaluate()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = new TestModel2($nn);
        $train_inputs = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $train_tests = $mo->array([0, 0, 0, 1, 1, 1],dtype:NDArray::int32);
        $val_inputs = $mo->array([[1, 2], [3, 2],]);
        $val_tests = $mo->array([0, 1,],dtype:NDArray::int32);

        $model->compile(
            loss:$nn->losses->SparseCategoricalCrossentropy(from_logits:true),
            optimizer:'adam',
        );

        $history = $model->fit(
            $train_inputs, $train_tests,
            batch_size:2,epochs:100,shuffle:true,verbose:0,
            validation_data:[$val_inputs,$val_tests]);

        if($this->plot) {
            $plt->plot($mo->array($history['loss']),null,null,'loss');
            $plt->plot($mo->array($history['accuracy']),null,null,'accuracy');
            $plt->plot($mo->array($history['val_loss']),null,null,'val_loss');
            $plt->plot($mo->array($history['val_accuracy']),null,null,'val_accuracy');
            $plt->legend();
            $plt->title('fit at dynamic mode with evaluation');
            $plt->show();
        }
        $this->assertTrue(true);
    }

    public function testNestModel()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = new Test3Main($nn);
        $train_inputs = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $train_tests = $mo->array([0, 0, 0, 1, 1, 1],dtype:NDArray::int32);
        $val_inputs = $mo->array([[1, 2], [3, 2],]);
        $val_tests = $mo->array([0, 1,],dtype:NDArray::int32);

        $model->compile(
            loss: $nn->losses->SparseCategoricalCrossentropy(from_logits:true),
            optimizer: 'adam',
        );

        $history = $model->fit(
            $train_inputs, $train_tests,
            batch_size:2,epochs:100,shuffle:true,verbose:0,
            validation_data:[$val_inputs,$val_tests]);

        if($this->plot) {
            $plt->plot($mo->array($history['loss']),null,null,'loss');
            $plt->plot($mo->array($history['accuracy']),null,null,'accuracy');
            $plt->plot($mo->array($history['val_loss']),null,null,'val_loss');
            $plt->plot($mo->array($history['val_accuracy']),null,null,'val_accuracy');
            $plt->legend();
            $plt->title('fit at dynamic mode with evaluation');
            $plt->show();
        }
        $this->assertTrue(true);
    }

    public function testSaveAndLoadLayers()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $model = new TestModel2($nn);
        $train_inputs = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $train_tests = $mo->array([0, 0, 0, 1, 1, 1],dtype:NDArray::int32);
        $val_inputs = $mo->array([[1, 2], [3, 2],]);
        $val_tests = $mo->array([0, 1,],dtype:NDArray::int32);

        $model->compile(
            loss: $nn->losses->SparseCategoricalCrossentropy(from_logits:true),
            optimizer: 'adam',
        );

        $history = $model->fit(
            $train_inputs, $train_tests,
            batch_size:2,epochs:100,shuffle:true,verbose:0,
            validation_data:[$val_inputs,$val_tests]);

        $savedWeights = [];
        $model->saveWeights($savedWeights);

        // =================================================================
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = new TestModel2($nn);
        $train_inputs = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $train_tests = $mo->array([0, 0, 0, 1, 1, 1],dtype:NDArray::int32);
        $val_inputs = $mo->array([[1, 2], [3, 2],]);
        $val_tests = $mo->array([0, 1,],dtype:NDArray::int32);

        $model->compile(
            loss: $nn->losses->SparseCategoricalCrossentropy(from_logits:true),
            optimizer: 'adam',
        );
        $model->loadWeights($savedWeights);

        $evals = $model->evaluate($val_inputs,$val_tests);

        $this->assertGreaterThanOrEqual(0.9,$evals['accuracy']);
        $this->assertLessThanOrEqual(0.5,$evals['loss']);
        //$model->summary();
    }

    public function testSaveAndLoadRNNLayers()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();

        $model = new TestRNNMain($mo,$nn);
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

        $history = $model->fit(
            $inputs, $targets,
            batch_size:2,epochs:1,shuffle:true,verbose:0);

        $savedWeights = [];
        $model->saveWeights($savedWeights);
        $weightShapes = array_map(fn($x)=>$x->shape(),$model->trainableVariables());

        // =================================================================
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = new TestRNNMain($mo,$nn);
        $inputs = $mo->array(
            [[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]],
            NDArray::int32
        );
        $targets = $mo->array(
            [[3, 1], [4, 1], [4, 2], [1, 3], [1, 4], [2, 4]],
            NDArray::int32
        );

        $model->compile(
            loss: 'sparse_categorical_crossentropy',
            optimizer: 'adam',
        );

        $model->loadWeights($savedWeights);
        $loadedShapes = array_map(fn($x)=>$x->shape(),$model->trainableVariables());
        $this->assertEquals($weightShapes,$loadedShapes);

        $seq = $mo->array([[1, 3, 4]],NDArray::int32);
        $attentionPlot = $mo->zeros([3, 3]);
        $results = $model->predict($seq,attention_plot:$attentionPlot);

        //$this->assertGreaterThanOrEqual(0.9,$totalAccuracy);
        //$this->assertLessThanOrEqual(0.4,$totalLoss);
        //$model->summary();
        $this->assertTrue(true);
    }

    public function testVariableModel()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = new TestVariableMain($nn);
        $lossfunc = $nn->losses->SparseCategoricalCrossentropy(from_logits:true);
        $train_inputs = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $train_tests = $mo->array([0, 0, 0, 1, 1, 1],dtype:NDArray::int32);
        $model->compile(
            loss: $lossfunc,
            optimizer: 'adam',
        );

        $history = $model->fit(
            $train_inputs, $train_tests,
            batch_size:2,epochs:100,shuffle:true,verbose:0);

        if($this->plot) {
            $plt->plot($mo->array($history['loss']),null,null,'loss');
            $plt->plot($mo->array($history['accuracy']),null,null,'accuracy');
            $plt->legend();
            $plt->title('function model');
            $plt->show();
        }

        $savedWeights = [];
        $model->saveWeights($savedWeights);
        //$origWeights = $model->trainableVariables();

        // =================================================================
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = new TestVariableMain($nn);
        $lossfunc = $nn->losses->SparseCategoricalCrossentropy(from_logits:true);
        $x = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $t = $mo->array([0, 0, 0, 1, 1, 1],dtype:NDArray::int32);

        $model->compile(
            loss: $lossfunc,
            optimizer: 'adam',
        );
        $model->loadWeights($savedWeights);

        $evals = $model->evaluate($x,$t);
        $this->assertLessThan(1.0,$evals['loss']);
        $this->assertEquals(1.0,$evals['accuracy']);
        //$model->summary();

        //$loadedWeights = $model->trainableVariables();
        //$this->assertCount(4,$loadedWeights);
        //foreach(array_map(null,$origWeights,$loadedWeights) as $d) {
        //    [$orig,$loaded] = $d;
        //    $this->assertTrue($mo->la()->isclose($orig->value(),$loaded->value()));
        //}
    }

    public function testGraphInModel()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = new TestGraphMode($nn);
        $lossfunc = $nn->losses->SparseCategoricalCrossentropy(from_logits:true);
        //$train_inputs = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        //$train_tests = $mo->array([0, 0, 0, 1, 1, 1]);
        $train_inputs = $mo->array([[1, 3]]);
        $train_tests = $mo->array([0],dtype:NDArray::int32);
        $model->compile(
            loss: $lossfunc,
            optimizer: 'adam',
        );
        $model->log('fit0');
        $history = $model->fit(
            $train_inputs, $train_tests,
            batch_size:2,epochs:1,shuffle:true,verbose:0);
        $model->log('fit1');
            $history = $model->fit(
            $train_inputs, $train_tests,
            batch_size:2,epochs:1,shuffle:true,verbose:0);
        $model->log('predict');
        $predicts = $model->predict($train_inputs);
        $model->log('invoke');
        $predicts = $model($g->Variable($train_inputs),true); // without graph

        $this->assertEquals([
            'fit0',
            'call',
            'fit1',
            'predict',
            'invoke',
            'call',
        ],$model->getLog());
    }

    public function testModelInGraph()
    {
        $mo = $this->newMatrixOperator();
        $nn = $this->newNeuralNetworks($mo);
        $K = $this->newBackend($nn);
        $g = $nn->gradient();
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = new TestGraphMode($nn);
        $lossfunc = $nn->losses->SparseCategoricalCrossentropy(from_logits:true);
        $train_inputs = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $train_tests = $mo->array([0, 0, 0, 1, 1, 1],dtype:NDArray::int32);
        $model->compile(
            loss: $lossfunc,
            optimizer: 'adam',
        );
        $model->log('fit0');
        $history = $model->fit(
            $train_inputs, $train_tests,
            batch_size:2,epochs:1,shuffle:true,verbose:0);
        $model->log('fit1');
            $history = $model->fit(
            $train_inputs, $train_tests,
            batch_size:2,epochs:1,shuffle:true,verbose:0);
        $model->log('predict');
        $predicts = $model->predict($train_inputs);

        $this->assertEquals([
            'fit0',
            'call',
            'fit1',
            'predict',
        ],$model->getLog());
    }
}
