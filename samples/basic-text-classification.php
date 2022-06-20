<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;

$mo = new MatrixOperator();
$nn = new NeuralNetworks($mo);
$plt = new Plot(null,$mo);

$datasetdir = sys_get_temp_dir().'/rindow/nn/datasets/aclImdb';
$tarfile = $datasetdir.'/aclImdb_v1.tar.gz';
$url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz";
$savefilename = __DIR__.'/aclImdb.pkl';
if(!file_exists($savefilename)) {
    if(!file_exists($datasetdir)) {
        mkdir($datasetdir,0777,true);
    }
    if(!file_exists($tarfile)) {
        echo "Loading....";
        copy($url, $tarfile);
        echo " Done.\n";
    }
    if(!file_exists($datasetdir.'/README')) {
        echo "Extract....";
        $phar = new PharData($tarfile);
        $rc=$phar->extractTo($datasetdir.'/..',null,true);
        echo " Done.\n";
    }
    $classnames = ['neg','pos'];
    $dataset = $nn->data->TextClassifiedDataset(
        $datasetdir.'/train',
        ['pattern'=>'@[0-9_]*\\.txt@','maxlen'=>256,'num_words'=>10000,
        'classnames'=>$classnames,'restricted_by_class'=>$classnames,
        'shuffle'=>true,'verbose'=>1]);
    [$train_inputs,$train_labels] = $dataset->loadData();
    $classnames = $dataset->classnames();
    $tokenizer = $dataset->getTokenizer();
    $dataset = $nn->data->TextClassifiedDataset(
        $datasetdir.'/test',
        ['pattern'=>'@[0-9_]*\\.txt@','tokenizer'=>$tokenizer,'maxlen'=>256,'num_words'=>10000,
        'classnames'=>$classnames,'restricted_by_class'=>$classnames,
        'shuffle'=>true,'verbose'=>1]);
    [$test_inputs,$test_labels] = $dataset->loadData();
    $train_labels = $mo->la()->astype($train_labels,NDArray::float32);
    $test_labels = $mo->la()->astype($test_labels,NDArray::float32);
    $savedata = [
        $tokenizer->save(),
        $classnames,
        $train_inputs,$train_labels,
        $test_inputs,$test_labels,
    ];
    file_put_contents($savefilename,serialize($savedata));
} else {
    $savedata =  unserialize(file_get_contents($savefilename));
    [
        $tokenizer_data,
        $classnames,
        $train_inputs,$train_labels,
        $test_inputs,$test_labels,
    ] = $savedata;
    $tokenizer = $nn->data->TextClassifiedDataset($datasetdir.'/train')
                    ->getTokenizer();
    $tokenizer->load($tokenizer_data);
}
$train_labels = $mo->la()->astype($train_labels,NDArray::float32);
$test_labels = $mo->la()->astype($test_labels,NDArray::float32);
echo implode(',',$train_inputs->shape())."\n";
echo implode(',',$train_labels->shape())."\n";
echo implode(',',$test_inputs->shape())."\n";
echo implode(',',$test_labels->shape())."\n";
$total_size = count($train_inputs);
$train_size = (int)floor($total_size*0.9);
$val_inputs = $train_inputs[[$train_size,$total_size-1]];
$val_labels = $train_labels[[$train_size,$total_size-1]];
$train_inputs = $train_inputs[[0,$train_size-1]];
$train_labels = $train_labels[[0,$train_size-1]];

$modelFilePath = __DIR__."/basic-text-classification.model";

if(file_exists($modelFilePath)) {
    echo "loading model ...\n";
    $model = $nn->models()->loadModel($modelFilePath);
    $model->summary();
} else {
    $inputlen = $train_inputs->shape()[1];
    echo "creating model ...\n";
    $model = $nn->models()->Sequential([
        $nn->layers()->Embedding(
            $inputDim=count($tokenizer->getWords()),
            $outputDim=16,
            ['input_length'=>$inputlen,]),
        $nn->layers()->GlobalAveragePooling1D(),
        $nn->layers()->Dense($units=1,
            ['activation'=>'sigmoid']),
    ]);

    $model->compile([
        'loss'=>$nn->losses->BinaryCrossEntropy(),
        'optimizer'=>'adam',
    ]);
    $model->summary();
    echo "training model ...\n";
    $history = $model->fit($train_inputs,$train_labels,[
        'epochs'=>10,'batch_size'=>64,
        'validation_data'=>[$val_inputs,$val_labels],
    ]);
    $model->save($modelFilePath);
    $plt->plot($mo->array($history['accuracy']),null,null,'accuracy');
    $plt->plot($mo->array($history['val_accuracy']),null,null,'val_accuracy');
    $plt->plot($mo->array($history['loss']),null,null,'loss');
    $plt->plot($mo->array($history['val_loss']),null,null,'val_loss');
    $plt->legend();
    $plt->title('imdb');
    $plt->show();
}
$model->evaluate($test_inputs,$test_labels,[
    'batch_size'=>64,'verbose'=>1,
]);
