<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;

$mo = new MatrixOperator();
$nn = new NeuralNetworks($mo);
$plt = new Plot(null,$mo);

if(!isset($argv[1])||!$argv[1]) {
    $shrink = false;
} else {
    $shrink = true;
}
$dataset='mnist';
if(isset($argv[2])) {
    $dataset=$argv[2];
}

if($dataset=='fashion') {
    [[$train_img,$train_label],[$test_img,$test_label]] =
        $nn->datasets()->fashionMnist()->loadData();
    $inputShape = [28,28,1];
    $shrinkEpochs = 3;
    $shrinkTrainSize = 5000;
    $shrinkTestSize  = 100;
} elseif($dataset=='cifar10') {
    [[$train_img,$train_label],[$test_img,$test_label]] =
        $nn->datasets()->cifar10()->loadData();
    $inputShape = [32,32,3];
    $shrinkEpochs = 3;
    $shrinkTrainSize = 4000;
    $shrinkTestSize  = 100;
} else {
    [[$train_img,$train_label],[$test_img,$test_label]] =
        $nn->datasets()->mnist()->loadData();
    $inputShape = [28,28,1];
    $shrinkEpochs = 3;
    $shrinkTrainSize = 5000;
    $shrinkTestSize  = 100;
}
$epochs = 5;

fwrite(STDERR,"train=[".implode(',',$train_img->shape())."]\n");
fwrite(STDERR,"test=[".implode(',',$test_img->shape())."]\n");

if($shrink||!extension_loaded('rindow_openblas')) {
    // Shrink data
    $epochs = $shrinkEpochs;
    $trainSize = $shrinkTrainSize;
    $testSize  = $shrinkTestSize;
    fwrite(STDERR,"Shrink data ...\n");
    $train_img = $train_img[[0,$trainSize-1]];
    $train_label = $train_label[[0,$trainSize-1]];
    $test_img = $test_img[[0,$testSize-1]];
    $test_label = $test_label[[0,$testSize-1]];
    fwrite(STDERR,"Shrink train=[".implode(',',$train_img->shape())."]\n");
    fwrite(STDERR,"Shrink test=[".implode(',',$test_img->shape())."]\n");
}


// flatten images and normalize
function formatingImage($mo,$train_img) {
    $dataSize = $train_img->shape()[0];
    $imageSize = $train_img[0]->size();
    $train_img = $train_img->reshape([$dataSize,$imageSize]);
    return $mo->scale(1.0/255.0,$mo->astype($train_img,NDArray::float32));
}

fwrite(STDERR,"formating train images ...\n");
$train_img = formatingImage($mo,$train_img);
fwrite(STDERR,"formating test images ...\n");
$test_img  = formatingImage($mo,$test_img);

[$dataSize,$imageSize] = $train_img->shape();
$train_img = $train_img->reshape(array_merge([$dataSize],$inputShape));
[$dataSize,$imageSize] = $test_img->shape();
$test_img = $test_img->reshape(array_merge([$dataSize],$inputShape));

fwrite(STDERR,"creating model ...\n");
$model = $nn->models()->Sequential([
    $nn->layers()->Conv2D(
       $filters=32,
        $kernel_size=3,
        ['input_shape'=>$inputShape,
        'kernel_initializer'=>'he_normal',
        'activation'=>'relu']),
    #$nn->layers()->MaxPooling2D(),
    $nn->layers()->AveragePooling2D(),
    $nn->layers()->Flatten(),
    $nn->layers()->Dense($units=128,
        ['kernel_initializer'=>'he_normal',
        'activation'=>'relu']),
    $nn->layers()->Dense($units=10,
         ['activation'=>'softmax']),
]);

$model->compile([
    'optimizer'=>$nn->optimizers()->Adam()
]);

fwrite(STDERR,"training model ...\n");
$history = $model->fit($train_img,$train_label,
    ['epochs'=>$epochs,'batch_size'=>256,'validation_data'=>[$test_img,$test_label]]);

$model->save(__DIR__.'/mnist-conv2d-model.model',$portable=true);

$model = $nn->models()->loadModel(__DIR__.'/mnist-conv2d-model.model');

$plt->plot($mo->array($history['accuracy']),null,null,'accuracy');
$plt->plot($mo->array($history['val_accuracy']),null,null,'val_accuracy');
$plt->plot($mo->array($history['loss']),null,null,'loss');
$plt->plot($mo->array($history['val_loss']),null,null,'val_loss');
$plt->legend();
$plt->title($dataset);
$plt->show();
