<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\NeuralNetworks\Support\GenericUtils;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Layer\AbstractRNNLayer;
use Rindow\NeuralNetworks\Model\AbstractModel;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

$TRAINING_SIZE = 20000;
$DIGITS = 3;
$REVERSE = True;
$WORD_VECTOR = 16;
$UNITS = 128;
$EPOCHS = 10;
$BATCH_SIZE = 8;


class NumAdditionDataset
{
    public function __construct($mo,int $corpus_max,int $digits)
    {
        $this->mo = $mo;
        $this->corpus_max = $corpus_max;
        $this->digits = $digits;
        #$this->reverse = $reverse;
        $this->vocab_input  = ['0','1','2','3','4','5','6','7','8','9','+',' '];
        $this->vocab_target = ['0','1','2','3','4','5','6','7','8','9','+',' '];
        $this->dict_input  = array_flip($this->vocab_input);
        $this->dict_target = array_flip($this->vocab_target);
        $this->input_length = $digits*2+1;
        $this->output_length = $digits+1;
    }

    public function dicts()
    {
        return [
            $this->vocab_input,
            $this->vocab_target,
            $this->dict_input,
            $this->dict_target,
        ];
    }

    public function generate()
    {
        $max_num = pow(10,$this->digits);
        $max_sample = $max_num ** 2;
        $numbers = $this->mo->random()->choice(
            $max_sample,$max_sample,$replace=false);
        $questions = [];
        $dups = [];
        $size = 0;
        for($i=0;$i<$max_sample;$i++) {
            $num = $numbers[$i];
            $x1 = (int)floor($num / $max_num);
            $x2 = (int)($num % $max_num);
            if($x1>$x2) {
                [$x1,$x2] = [$x2,$x1];
            }
            $question = $x1.'+'.$x2;
            if(array_key_exists($question,$questions)) {
                #echo $question.',';
                $dups[$question] += 1;
                continue;
            }
            $dups[$question] = 1;
            $questions[$question] = strval($x1+$x2);
            $size++;
            if($size >= $this->corpus_max)
                break;
        }
        unset($numbers);
        $sequence = $this->mo->zeros([$size,$this->input_length],NDArray::int32);
        $target = $this->mo->zeros([$size,$this->output_length],NDArray::int32);
        $i = 0;
        foreach($questions as $question=>$answer) {
            $question = str_pad($question, $this->input_length);
            $answer = str_pad($answer, $this->output_length);
            $this->str2seq(
                $question,
                $this->dict_input,
                $sequence[$i]);
            $this->str2seq(
                $answer,
                $this->dict_target,
                $target[$i]);
            $i++;
        }
        return [$sequence,$target];
    }

    public function str2seq(
        string $str,
        array $dic,
        NDArray $buf)
    {
        $sseq = str_split(strtoupper($str));
        $len = count($sseq);
        $sp = $dic[' '];
        $bufsz=$buf->size();
        for($i=0;$i<$bufsz;$i++){
            if($i<$len)
                $buf[$i]=$dic[$sseq[$i]];
            else
                $buf[$i]=$sp;
        }
    }

    public function seq2str(
        NDArray $buf,
        array $dic
        )
    {
        $str = '';
        $bufsz=$buf->size();
        for($i=0;$i<$bufsz;$i++){
            $str .= $dic[$buf[$i]];
        }
        return $str;
    }

    public function loadData($path=null)
    {
        if($path==null){
            $path='numaddition-dataset.pkl';
        }
        if(file_exists($path)){
            $pkl = file_get_contents($path);
            $dataset = unserialize($pkl);
        }else{
            $dataset = $this->generate();
            $pkl = serialize($dataset);
            file_put_contents($path,$pkl);
        }
        return $dataset;
    }
}

$mo = new MatrixOperator();
$nn = new NeuralNetworks($mo);
$plt = new Plot(null,$mo);

$input_length  = $DIGITS*2 + 1;
$output_length = $DIGITS + 1;

$dataset = new NumAdditionDataset($mo,$TRAINING_SIZE,$DIGITS);
echo "Generating data...\n";
[$questions,$answers] = $dataset->loadData();
$corpus_size = $questions->shape()[0];
echo "Total questions: ". $corpus_size."\n";
[$input_voc,$target_voc,$input_dic,$target_dic]=$dataset->dicts();


# Explicitly set apart 10% for validation data that we never train over.
$split_at = $corpus_size - (int)floor($corpus_size / 10);
$x_train = $questions[[0,$split_at-1]];
$x_val   = $questions[[$split_at,$corpus_size-1]];
$y_train = $answers[[0,$split_at-1]];
$y_val   = $answers[[$split_at,$corpus_size-1]];

echo "train,test: ".$x_train->shape()[0].",".$y_train->shape()[0]."\n";

$modelFilePath = __DIR__."/learn-add-numbers-with-rnn.model";

if(file_exists($modelFilePath)) {
    echo "loading model ...\n";
    $model = $nn->models()->loadModel($modelFilePath);
    $model->summary();
} else {
    echo "Build model...\n";

    $model = $nn->models()->Sequential([
        $nn->layers()->Embedding(count($input_dic), $WORD_VECTOR,
            ['input_length'=>$input_length]
        ),
        # Encoder
        $nn->layers()->GRU($UNITS,['go_backwards'=>$REVERSE]),
        # Expand to answer length and peeking hidden states
        $nn->layers()->RepeatVector($output_length),
        # Decoder
        $nn->layers()->GRU($UNITS, [
            'return_sequences'=>true,
            'go_backwards'=>$REVERSE,
            #'reset_after'=>false,
        ]),
        # Output
        $nn->layers()->Dense(
            count($target_dic),
            ['activation'=>'softmax']
        ),
    ]);

    echo "Compile model...\n";

    $model->compile([
        'loss'=>'sparse_categorical_crossentropy',
        'optimizer'=>'adam',
        ]);
    $model->summary();

    # Train the model
    echo "Train model...\n";

    $history = $model->fit(
        $x_train,
        $y_train,
        [
            'epochs'=>$EPOCHS,
            'batch_size'=>$BATCH_SIZE,
            'validation_data'=>[$x_val, $y_val],
        ]
    );

    $model->save($modelFilePath);

    $plt->plot($mo->array($history['accuracy']),null,null,'accuracy');
    $plt->plot($mo->array($history['val_accuracy']),null,null,'val_accuracy');
    $plt->plot($mo->array($history['loss']),null,null,'loss');
    $plt->plot($mo->array($history['val_loss']),null,null,'val_loss');
    $plt->legend();
    $plt->title('seq2seq-simple-numaddition');
    $plt->show();
}

for($i=0;$i<10;$i++) {
    $idx = $mo->random()->randomInt($corpus_size);
    $question = $questions[$idx];
    $input = $question->reshape([1,$input_length]);

    $predict = $model->predict($input);
    $predict_seq = $mo->argMax($predict[0]->reshape([$output_length,count($target_dic)]),$axis=1);
    $predict_str = $dataset->seq2str($predict_seq,$target_voc);
    $question_str = $dataset->seq2str($question,$input_voc);
    $answer_str = $dataset->seq2str($answers[$idx],$target_voc);
    $correct = ($predict_str==$answer_str) ? '*' : ' ';
    echo "$question_str=$predict_str : $correct $answer_str\n";
}
