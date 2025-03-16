<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Data\Dataset\DatasetFilter;
use Interop\Polite\Math\Matrix\NDArray;
use function Rindow\Math\Matrix\R;

$mo = new MatrixOperator();
$nn = new NeuralNetworks($mo);
$plt = new Plot(null,$mo);
//$nn->backend()->primaryLA()->setProfiling(true);


$dsname='mnist';
$epochs = 5;
$batch_size = 256;
$shrink = false;


if(isset($argv[1])&&$argv[1]) {
    $dsname=$argv[1];
}
if(isset($argv[2])&&$argv[2]) {
    $epochs = $argv[2];
}
if(isset($argv[3])&&$argv[3]) {
    $shrink = true;
}

echo "dataset={$dsname}\n";
switch($dsname) {
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
        [$train_data,$test_data] = $nn->datasets()->cifar10()->loadData();
        $inputShape = [32,32,3];
        $class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck'];
        break;
    }
    default: {
        echo "Unknown dataset $dsname\n";
        exit(1);
    }
}

$inputsFilter = new class ($mo,$inputShape) implements DatasetFilter
{
    protected object $mo;
    protected array $inputShape;
    public function __construct(object $mo,array $inputShape) {
        $this->mo = $mo;
        $this->inputShape = $inputShape;
    }
    public function translate(
        iterable $img,
        ?iterable $label=null,
        ?array $options=null) : array
    {
        $la = $this->mo->la();
        $dataSize = $img->shape()[0];
        $imageSize = $img[0]->size();
        $shape = array_merge([$dataSize],$this->inputShape);
        $img = $img->reshape($shape);
        return [
            $this->mo->scale(1.0/255.0,$this->mo->astype($img,NDArray::float32)),
            $this->mo->la()->astype($label,NDArray::int32),
        ];
    }
};


switch($dsname) {
    case 'mnist': 
    case 'fashion': {
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
        
        echo "formating train images and labels ...\n";
        [$train_img,$train_label] = $inputsFilter->translate($train_img,$train_label);
        [$test_img,$test_label] = $inputsFilter->translate($test_img,$test_label);

        echo "train=[".$mo->shapeToString($train_img->shape()).",".
                        $mo->shapeToString($train_label->shape())."]\n";
        echo "val=[".$mo->shapeToString($test_img->shape()).",".
                        $mo->shapeToString($test_label->shape())."]\n";
        $dataset = $nn->data->NDArrayDataset(
            $train_img,
            tests:$train_label,
            batch_size:$batch_size,
        );
        $val_dataset = $nn->data->NDArrayDataset(
            $test_img,
            tests:$test_label,
            batch_size:$batch_size,
        );

        break;
    }
    case 'cifar10': {
        $trainSize = 50000;
        $testSize = 10000;
        if($shrink||!$mo->isAdvanced()) {
            // Shrink data
            $trainSize = 2000;
            $testSize  = 200;
            echo "Shrink data ...\n";
            echo "Shrink train=$trainSize\n";
            echo "Shrink test=$testSize\n";
        }
        echo "train_size=$trainSize\n";
        echo "val_size=$testSize\n";
        echo "formating train images and labels ...\n";
        $dataset = $nn->data->SequentialDataset(
            $train_data,
            batch_size:$batch_size,
            inputs_filter:$inputsFilter,
            total_size:$trainSize,
        );
        $val_dataset = $nn->data->SequentialDataset(
            $test_data,
            batch_size:$batch_size,
            inputs_filter:$inputsFilter,
            total_size:$testSize,
        );
        break;
    }
    default: {
        echo "Unknown dataset $dsname\n";
        exit(1);
    }
}


echo "device type: ".$nn->deviceType()."\n";
$modelFilePath = __DIR__."/cnn-image-classification-{$dsname}.model";

if(file_exists($modelFilePath)) {
    echo "loading model ...\n";
    $model = $nn->models()->loadModel($modelFilePath);
    $model->summary();
} else {
    echo "creating model ...\n";
    $model = $nn->models()->Sequential([
        $nn->layers()->Conv2D(
            $filters=32,
            $kernel_size=3,
            input_shape:$inputShape,
            kernel_initializer:'he_normal',),
        $nn->layers()->MaxPooling2D(),
        $nn->layers()->Conv2D(
            $filters=64,
            $kernel_size=3,
            kernel_initializer:'he_normal',),
        $nn->layers()->MaxPooling2D(),
        $nn->layers()->Conv2D(
            $filters=128,
            $kernel_size=3,
            kernel_initializer:'he_normal',),
        $nn->layers()->GlobalMaxPooling2D(),
        $nn->layers()->Dense($units=128,
            kernel_initializer:'he_normal',
            activation:'relu',
        ),
        $nn->layers()->Dense($units=10
            /* ,activation:'softmax' */),
    ]);

    $model->compile(
        loss:$nn->losses()->SparseCategoricalCrossentropy(from_logits:true),
        optimizer:'adam',
    );
    $model->summary();
    echo "training model ...\n";
    $history = $model->fit(
        $dataset,
        epochs:$epochs,
        validation_data:$val_dataset);
    $model->save($modelFilePath,$portable=true);
    $plt->plot($mo->array($history['accuracy']),null,null,'accuracy');
    $plt->plot($mo->array($history['val_accuracy']),null,null,'val_accuracy');
    $plt->plot($mo->array($history['loss']),null,null,'loss');
    $plt->plot($mo->array($history['val_loss']),null,null,'val_loss');
    $plt->legend();
    $plt->title($dsname);
}

foreach($val_dataset as $idx => [$img,$lbl]) {
    $images = $img;
    $labels = $lbl;
}
$images = $images[R(0,8)];
$labels = $labels[R(0,8)];
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

//$nn->backend()->primaryLA()->profilingReport();