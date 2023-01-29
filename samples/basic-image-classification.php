<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;

$mo = new MatrixOperator();
$nn = new NeuralNetworks($mo);
$plt = new Plot(null,$mo);


$dataset='mnist';
$epochs = 5;
$shrink = false;


if(isset($argv[1])&&$argv[1]) {
    $dataset=$argv[1];
}
if(isset($argv[2])&&$argv[2]) {
    $epochs = $argv[3];
}
if(isset($argv[3])&&$argv[3]) {
    $shrink = true;
}

switch($dataset) {
    case 'mnist': {
        [[$train_img,$train_label],[$test_img,$test_label]] =
            $nn->datasets()->mnist()->loadData();
        $inputShape = [28,28,1];
        $class_names = [0,1,2,3,4,5,6,7,8,9];
        break;
    }
    case 'fashion': {
        [[$train_img,$train_label],[$test_img,$test_label]] =
            $nn->datasets()->fashionMnist()->loadData();
        $inputShape = [28,28,1];
        $class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'];
        break;
    }
    case 'cifar10': {
        [[$train_img,$train_label],[$test_img,$test_label]] =
            $nn->datasets()->cifar10()->loadData();
        $inputShape = [32,32,3];
        $class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck'];
        break;
    }
    default: {
        echo "Unknown dataset $dataset\n";
        exit(1);
    }
}
echo "dataset={$dataset}\n";
echo "train=[".implode(',',$train_img->shape())."]\n";
echo "test=[".implode(',',$test_img->shape())."]\n";

if($shrink||!extension_loaded('rindow_openblas')) {
    // Shrink data
    $trainSize = 2000;
    $testSize  = 200;
    echo "Shrink data ...\n";
    $train_img = $train_img[[0,$trainSize-1]];
    $train_label = $train_label[[0,$trainSize-1]];
    $test_img = $test_img[[0,$testSize-1]];
    $test_label = $test_label[[0,$testSize-1]];
    echo "Shrink train=[".implode(',',$train_img->shape())."]\n";
    echo "Shrink test=[".implode(',',$test_img->shape())."]\n";
}

// flatten images and normalize
function formatingImage($mo,$train_img) {
    $dataSize = $train_img->shape()[0];
    $imageSize = $train_img[0]->size();
    $train_img = $train_img->reshape([$dataSize,$imageSize]);
    return $mo->scale(1.0/255.0,$mo->astype($train_img,NDArray::float32));
}

//echo "slice images ...\n";
//$samples = 1000;
//$testSamples = (int)min(ceil($samples/10),count($test_img));
//$train_img = $train_img[[0,$samples-1]];
//$train_label = $train_label[[0,$samples-1]];
//$test_img = $test_img[[0,$testSamples-1]];
//$test_label = $test_label[[0,$testSamples-1]];
//echo "Truncated train=[".implode(',',$train_img->shape())."]\n";
//echo "Truncated test=[".implode(',',$test_img->shape())."]\n";


echo "formating train images ...\n";
$train_img = formatingImage($mo,$train_img);
$train_label = $mo->la()->astype($train_label,NDArray::int32);
echo "formating test images ...\n";
$test_img  = formatingImage($mo,$test_img);
$test_label = $mo->la()->astype($test_label,NDArray::int32);

$modelFilePath = __DIR__."/basic-image-classification-{$dataset}.model";

if(file_exists($modelFilePath)) {
    echo "loading model ...\n";
    $model = $nn->models()->loadModel($modelFilePath);
    $model->summary();
} else {
    echo "creating model ...\n";
    $model = $nn->models()->Sequential([
        $nn->layers()->Dense($units=128,
            input_shape:[(int)array_product($inputShape)],
            kernel_initializer:'he_normal',
            activation:'relu',
            ),
        $nn->layers()->Dense($units=10
            /*,['activation'=>'softmax']*/),
    ]);

    $model->compile(
        loss:$nn->losses()->SparseCategoricalCrossentropy(from_logits:true),
        optimizer:'adam',
    );
    $model->summary();
    echo "training model ...\n";
    $history = $model->fit($train_img,$train_label,
        epochs:5,batch_size:256,validation_data:[$test_img,$test_label]);
    $model->save($modelFilePath,$portable=true);
    $plt->plot($mo->array($history['accuracy']),null,null,'accuracy');
    $plt->plot($mo->array($history['val_accuracy']),null,null,'val_accuracy');
    $plt->plot($mo->array($history['loss']),null,null,'loss');
    $plt->plot($mo->array($history['val_loss']),null,null,'val_loss');
    $plt->legend();
    $plt->title($dataset);
}

$images = $test_img[[0,7]];
$labels = $test_label[[0,7]];
$predicts = $model->predict($images);
// for from_logits
$K = $nn->backend();
$predicts = $K->ndarray($nn->backend->softmax($K->array($predicts)));

if($inputShape[2]==1) {
    array_pop($inputShape);
}
$plt->setConfig([
    'frame.xTickLength'=>0,'title.position'=>'down','title.margin'=>0,]);
[$fig,$axes] = $plt->subplots(4,4);
foreach ($predicts as $i => $predict) {
    $axes[$i*2]->imshow($images[$i]->reshape($inputShape),
        null,null,null,$origin='upper');
    $axes[$i*2]->setFrame(false);
    $label = $labels[$i];
    $axes[$i*2]->setTitle($class_names[$label]."($label)");
    $axes[$i*2+1]->bar($mo->arange(10),$predict);
}

$plt->show();
