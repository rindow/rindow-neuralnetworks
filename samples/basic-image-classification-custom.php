<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use function Rindow\Math\Matrix\R;

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

[[$train_img,$train_label],[$test_img,$test_label]] =
    $nn->datasets()->mnist()->loadData();
$inputShape = [28,28,1];
$class_names = [0,1,2,3,4,5,6,7,8,9];

echo "dataset={$dataset}\n";
echo "train=[".implode(',',$train_img->shape())."]\n";
echo "test=[".implode(',',$test_img->shape())."]\n";

if($shrink||!$mo->isAdvanced()) {
    // Shrink data
    $trainSize = 2000;
    $testSize  = 200;
    echo "Shrink data ...\n";
    $train_img = $train_img[R(0,$trainSize)];
    $train_label = $train_label[R(0,$trainSize)];
    $test_img = $test_img[R(0,$testSize)];
    $test_label = $test_label[R(0,$testSize)];
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

echo "formating train images ...\n";
$train_img = formatingImage($mo,$train_img);
$train_label = $mo->la()->astype($train_label,NDArray::int32);
echo "formating test images ...\n";
$test_img  = formatingImage($mo,$test_img);
$test_label = $mo->la()->astype($test_label,NDArray::int32);

use Rindow\NeuralNetworks\Model\AbstractModel;
class ImageClassification extends AbstractModel
{
    protected $dense1;
    protected $dense2;
    public function __construct(
        $builder
        )
    {
        parent::__construct($builder);
        $this->dense1 = $builder->layers->Dense($units=128,
            input_shape:[(int)array_product([28,28,1])],
            kernel_initializer:'he_normal',
            activation:'relu',
        );
        $this->dense2 = $builder->layers->Dense($units=10);
    }

    protected function call($inputs)
    {
        $x = $this->dense1->forward($inputs);
        $outputs = $this->dense2->forward($x);
        return $outputs;
    }
}

echo "device type: ".$nn->deviceType()."\n";
$model = new ImageClassification($nn);
echo "creating model ...\n";
$model->compile(
    loss:$nn->losses()->SparseCategoricalCrossentropy(from_logits:true),
    optimizer:'adam',
);
$model->build([1,(int)(28*28*1)]); // This is only needed for summary
$model->summary();

$modelFilePath = __DIR__."/basic-image-classification-custom-{$dataset}.model";

if(file_exists($modelFilePath)) {
    echo "loading model ...\n";
    $model->loadWeightsFromFile($modelFilePath,$portable=true);
} else {
    echo "training model ...\n";
    $history = $model->fit($train_img,$train_label,
        epochs:5,batch_size:256,validation_data:[$test_img,$test_label]);
    $model->saveWeightsToFile($modelFilePath,$portable=true);
    $plt->plot($mo->array($history['accuracy']),null,null,'accuracy');
    $plt->plot($mo->array($history['val_accuracy']),null,null,'val_accuracy');
    $plt->plot($mo->array($history['loss']),null,null,'loss');
    $plt->plot($mo->array($history['val_loss']),null,null,'val_loss');
    $plt->legend();
    $plt->title($dataset);
}

$images = $test_img[R(0,8)];
$labels = $test_label[R(0,8)];
$predicts = $model->predict($images);
// for from_logits
$predicts = $nn->backend->array($predicts);
$predicts = $nn->backend->softmax($predicts);
$predicts = $nn->backend->ndarray($predicts);

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
