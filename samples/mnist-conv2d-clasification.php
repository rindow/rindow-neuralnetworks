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
if(!isset($argv[2])||!$argv[2]) {
    [[$train_img,$train_label],[$test_img,$test_label]] =
        $nn->datasets()->mnist()->loadData();
} else {
    [[$train_img,$train_label],[$test_img,$test_label]] =
        $nn->datasets()->fashionMnist()->loadData();
}
$epochs = 5;

fwrite(STDERR,"train=[".implode(',',$train_img->shape())."]\n");
fwrite(STDERR,"test=[".implode(',',$test_img->shape())."]\n");

if($shrink||!extension_loaded('rindow_openblas')) {
    // Shrink data
    $epochs = 5;
    $trainSize = 6000;
    $testSize  = 100;
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
$train_img = $train_img->reshape([$dataSize,28,28,1]);
[$dataSize,$imageSize] = $test_img->shape();
$test_img = $test_img->reshape([$dataSize,28,28,1]);

fwrite(STDERR,"creating model ...\n");
$model = $nn->models()->Sequential([
    $nn->layers()->Conv2D(
       $filters=30,
        $kernel_size=3,
        ['input_shape'=>[28,28,1],
        'kernel_initializer'=>'relu_normal']),
    $nn->layers()->ReLU(),
    #$nn->layers()->MaxPooling2D(),
    $nn->layers()->AveragePooling2D(),
    $nn->layers()->Flatten(),
    $nn->layers()->Dense($units=100,
        ['kernel_initializer'=>'relu_normal']),
    $nn->layers()->ReLU(),
    $nn->layers()->Dense($units=10),
    $nn->layers()->Softmax(),
]);

$model->compile([
    'optimizer'=>$nn->optimizers()->Adam()
]);

fwrite(STDERR,"training model ...\n");
$history = $model->fit($train_img,$train_label,
    ['epochs'=>$epochs,'batch_size'=>256,'validation_data'=>[$test_img,$test_label]]);

$model->save(__DIR__.'/mnist_conv_model.model',$portable=true);

$model = $nn->models()->loadModel(__DIR__.'/mnist_conv_model.model');

$plt->plot($mo->array($history['accuracy']),null,null,'accuracy');
$plt->plot($mo->array($history['val_accuracy']),null,null,'val_accuracy');
$plt->plot($mo->array($history['loss']),null,null,'loss');
$plt->plot($mo->array($history['val_loss']),null,null,'val_loss');
$plt->legend();
$plt->title('mnist');
$plt->show();
