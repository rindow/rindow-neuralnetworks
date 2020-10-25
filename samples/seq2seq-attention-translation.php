<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\NeuralNetworks\Support\GenericUtils;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Layer\AbstractRNNLayer;
use Rindow\NeuralNetworks\Model\AbstractModel;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Data\Sequence\Tokenizer;
use Rindow\NeuralNetworks\Data\Sequence\Preprocessor;

# Download the file
class EngFraDataset
{
    protected $baseUrl = 'http://www.manythings.org/anki/';
    protected $downloadFile = 'fra-eng.zip';

    public function __construct($mo,$inputTokenizer=null,$targetTokenizer=null)
    {
        $this->mo = $mo;
        $this->datasetDir = $this->getDatasetDir();
        if(!file_exists($this->datasetDir)) {
            @mkdir($this->datasetDir,0777,true);
        }
        $this->saveFile = $this->datasetDir . "/fra-eng.pkl";
        $this->preprocessor = new Preprocessor($mo);
    }

    protected function getDatasetDir()
    {
        return sys_get_temp_dir().'/rindow/nn/datasets/fra-eng';
    }

    protected function download($filename)
    {
        $filePath = $this->datasetDir . "/" . $filename;

        if(!file_exists($filePath)){
            $this->console("Downloading " . $filename . " ... ");
            copy($this->baseUrl.$filename, $filePath);
            $this->console("Done\n");
        }

        $memberfile = 'fra.txt';
        $path = $this->datasetDir.'/'.$memberfile;
        if(file_exists($path)){
            return $path;
        }
        $this->console("Extract to:".$this->datasetDir.'/..'."\n");
        $files = [$memberfile];
        $zip = new ZipArchive();
        $zip->open($filePath);
        $zip->extractTo($this->datasetDir);
        $zip->close();
        $this->console("Done\n");

        return $path;
    }

    # Converts the unicode file to ascii
    #def unicode_to_ascii(self,s):
    #    return ''.join(c for c in unicodedata.normalize('NFD', s)
    #    if unicodedata.category(c) != 'Mn')

    public function preprocessSentence($w)
    {
        $w = '<start> '.$w.' <end>';
        return $w;
    }

    public function createDataset($path, $numExamples)
    {
        $contents = file_get_contents($path);
        if($contents==false) {
            throw new InvalidArgumentException('file not found: '.$path);
        }
        $lines = explode("\n",trim($contents));
        unset($contents);
        $trim = function($w) { return trim($w); };
        $enSentences = [];
        $spSentences = [];
        foreach ($lines as $line) {
            if($numExamples!==null) {
                $numExamples--;
                if($numExamples<0)
                    break;
            }
            $blocks = explode("\t",$line);
            $blocks = array_map($trim,$blocks);
            $en = $this->preprocessSentence($blocks[0]);
            $sp = $this->preprocessSentence($blocks[1]);
            $enSentences[] = $en;
            $spSentences[] = $sp;
        }
        return [$enSentences,$spSentences];
    }

    public function tokenize($lang,$numWords=null,$tokenizer=null)
    {
        if($tokenizer==null) {
            $tokenizer = new Tokenizer($this->mo,[
                'num_words'=>$numWords,
                'filters'=>"\"\'#$%&()*+,-./:;=@[\\]^_`{|}~\t\n",
                'specials'=>"?.!,¿",
            ]);
        }
        $tokenizer->fitOnTexts($lang);
        $sequences = $tokenizer->textsToSequences($lang);
        $tensor = $this->preprocessor->padSequences($sequences,['padding'=>'post']);
        return [$tensor, $tokenizer];
    }

    protected function console($message)
    {
        fwrite(STDERR,$message);
    }

    public function loadData(
        string $path=null, int $numExamples=null, int $numWords=null)
    {
        if($path==null) {
            $path = $this->download($this->downloadFile);
        }
        # creating cleaned input, output pairs
        [$targ_lang, $inp_lang] = $this->createDataset($path, $numExamples);

        [$input_tensor, $inp_lang_tokenizer] = $this->tokenize($inp_lang,$numWords);
        [$target_tensor, $targ_lang_tokenizer] = $this->tokenize($targ_lang,$numWords);
        $numInput = $input_tensor->shape()[0];
        $choice = $this->mo->random()->choice($numInput,$numInput,$replace=false);
        $input_tensor = $this->shuffle($input_tensor,$choice);
        $target_tensor = $this->shuffle($target_tensor,$choice);

        return [$input_tensor, $target_tensor, $inp_lang_tokenizer, $targ_lang_tokenizer];
    }

    public function shuffle(NDArray $tensor, NDArray $choice) : NDArray
    {
        $result = $this->mo->zerosLike($tensor);
        $size = $tensor->shape()[0];
        for($i=0;$i<$size;$i++) {
            $this->mo->la()->copy($tensor[$choice[$i]],$result[$i]);
        }
        return $result;
    }

    public function convert($lang, NDArray $tensor) : void
    {
        $size = $tensor->shape()[0];
        for($i=0;$t<$size;$t++) {
            $t = $tensor[$i];
            if($t!=0)
                echo sprintf("%d ----> %s\n", $t, $lang->index_word[$t]);
        }
    }
}

class Encoder extends AbstractRNNLayer
{
    public function __construct(
        $backend,
        $builder,
        int $vocabSize,
        int $wordVectSize,
        int $units
        )
    {
        $this->vocabSize = $vocabSize;
        $this->wordVectSize = $wordVectSize;
        $this->units = $units;
        $this->embedding = $builder->layers()->Embedding($vocabSize,$wordVectSize);
        $this->rnn = $builder->layers()->GRU(
            $units,
            ['return_state'=>true,'return_sequences'=>true,
             'recurrent_initializer'=>'glorot_uniform']
        );
    }

    public function build(array $inputShape=null, array $options=null) : array
    {
        $inputShape = $this->normalizeInputShape($inputShape);
        $inputShape = $this->registerLayer($this->embedding,$inputShape);
        [$outputShape,$statesShapes] = $this->registerLayer($this->rnn,$inputShape);
        $this->outputShape = $outputShape;
        $this->statesShapes = $statesShapes;
        return [$outputShape,$statesShapes];
    }

    public function getConfig() : array
    {
        return [
            'builder'=>true,
            'vocab_size'=>$this->vocabSize,
            'word_vec_size'=>$this->wordVecSize,
            'units'=>$this->units,
            ];
    }

    protected function call(
        NDArray $inputs,
        bool $training,
        array $initial_state=null,
        array $options=null
        ) : array
    {
        $wordVect = $this->embedding->forward($inputs,$training);
        [$outputs,$states] = $this->rnn->forward(
            $wordVect,$training,$initial_state);
        return [$outputs, $states];
    }

    protected function differentiate(NDArray $dOutputs, array $dStates=null)
    {
        [$dWordvect,$dStates] = $this->rnn->backward($dOutputs,$dStates);
        $dInputs = $this->embedding->backward($dWordvect);
        return $dInputs;
    }

    public function initializeHiddenState($batch_sz)
    {
        return $this->backend->zeros([$batch_sz, $this->units]);
    }
}

class Decoder extends AbstractRNNLayer
{
    protected $backend;
    protected $vocabSize;
    protected $wordVectSize;
    protected $units;
    protected $targetLength;
    protected $embedding;
    protected $rnn;
    protected $attention;
    protected $concat;
    protected $dense;

    public function __construct(
        $backend,
        $builder,
        int $vocabSize,
        int $wordVectSize,
        int $units,
        int $inputLength,
        int $targetLength
        )
    {
        $this->backend = $backend;
        $this->vocabSize = $vocabSize;
        $this->wordVectSize = $wordVectSize;
        $this->units = $units;
        $this->inputLength = $inputLength;
        $this->targetLength = $targetLength;
        $this->embedding = $builder->layers()->Embedding($vocabSize, $wordVectSize);
        $this->rnn = $builder->layers()->GRU($units,
            ['return_state'=>true,'return_sequences'=>true,
             'recurrent_initializer'=>'glorot_uniform']
        );
        $this->attention = $builder->layers()->Attention(
            ['return_attention_scores'=>true]);
        $this->concat = $builder->layers()->Concatenate();
        $this->dense = $builder->layers()->Dense($vocabSize);
    }

    public function build(array $inputShape=null, array $options=null) : array
    {
        $encOutputsShape = [$this->inputLength,$this->units];
        $inputShape = $this->normalizeInputShape($inputShape);
        $inputShape = $this->registerLayer($this->embedding,$inputShape);
        [$rnnShape,$statesShapes] = $this->registerLayer($this->rnn,$inputShape);

        [$contextVectorShape,$scoresShape] = $this->registerLayer($this->attention,
            [$rnnShape,$encOutputsShape]);

        $outputShape = $this->registerLayer($this->concat,[$contextVectorShape,$rnnShape]);
        $outputShape = $this->registerLayer($this->dense,$outputShape);
        $this->outputShape = $outputShape;
        $this->statesShapes = $statesShapes;
        return [$outputShape,$statesShapes];
    }

    public function getConfig() : array
    {
        return [
            'builder'=>true,
            'rnn'=>$this->rnnName,
            'vocab_size'=>$this->vocabSize,
            'word_vec_size'=>$this->wordVecSize,
            'units'=>$this->units,
        ];
    }

    protected function call(
        NDArray $inputs,
        bool $training,
        array $initial_state=null,
        array $options=null
        ) : array
    {
        $K = $this->backend;
        $encOutputs=$options['enc_outputs'];

        $x = $this->embedding->forward($inputs,$training);
        [$rnnSequence,$states] = $this->rnn->forward(
            $x,$training,$initial_state);

        [$contextVector,$attentionScores] = $this->attention->forward(
            [$rnnSequence,$encOutputs],$training);
        $outputs = $this->concat->forward([$contextVector, $rnnSequence],$training);

        $outputs = $this->dense->forward($outputs,$training);
        $this->contextVectorShape = $contextVector->shape();
        $this->rnnSequenceShape = $rnnSequence->shape();
        $this->attentionScores = $attentionScores;
        return [$outputs,$states];
    }

    public function getAttentionScores()
    {
        return $this->attentionScores;
    }

    protected function differentiate(NDArray $dOutputs, array $dNextStates=null)
    {
        $K = $this->backend;
        $dOutputs = $this->dense->backward($dOutputs);
        [$dContextVector,$dRnnSequence] = $this->concat->backward($dOutputs);
        [$dRnnSequence2,$dEncOutputs] = $this->attention->backward($dContextVector);
        $K->update_add($dRnnSequence,$dRnnSequence2);
        [$dWordVect,$dStates]=$this->rnn->backward($dRnnSequence,$dNextStates);
        $dInputs = $this->embedding->backward($dWordVect);
        return [$dInputs,$dStates,['enc_outputs'=>$dEncOutputs]];
    }
}


class Seq2seq extends AbstractModel
{
    public function __construct(
        $mo,
        $backend,
        $builder,
        $inputLength=null,
        $inputVocabSize=null,
        $outputLength=null,
        $targetVocabSize=null,
        $wordVectSize=8,
        $units=256,
        $startVocId=0,
        $endVocId=0,
        $plt
        )
    {
        parent::__construct($backend,$builder);
        $this->encoder = new Encoder(
            $backend,
            $builder,
            $inputVocabSize,
            $wordVectSize,
            $units
        );
        $this->decoder = new Decoder(
            $backend,
            $builder,
            $targetVocabSize,
            $wordVectSize,
            $units,
            $inputLength,
            $outputLength
        );
        $this->out = $builder->layers()->Activation('softmax');
        $this->setLastLayer($this->out);
        $this->mo = $mo;
        $this->backend = $backend;
        $this->startVocId = $startVocId;
        $this->endVocId = $endVocId;
        $this->inputLength = $inputLength;
        $this->outputLength = $outputLength;
        $this->units = $units;
        $this->plt = $plt;
    }

    protected function buildLayers(array $options=null) : void
    {
        $shape = [$this->inputLength];
        [$encOutputsShape,$encStatesShapes] = $this->registerLayer($this->encoder,$shape);
        $shape = [$this->outputLength];
        [$outputsShape,$statesShapes] = $this->registerLayer($this->decoder,$shape);
        $this->registerLayer($this->out,$outputsShape);
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

    protected function forwardStep(NDArray $inputs, NDArray $trues=null, bool $training=null) : NDArray
    {
        $K = $this->backend;
        [$encOutputs,$states] = $this->encoder->forward($inputs,$training);
        $options = ['enc_outputs'=>$encOutputs];
        [$outputs,$dummy] = $this->decoder->forward($trues,$training,$states,$options);
        $outputs = $this->out->forward($outputs,$training);
        return $outputs;
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

    protected function backwardStep(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        $dOutputs = $this->out->backward($dOutputs);
        [$dummy,$dStates,$dOptions] = $this->decoder->backward($dOutputs,null);
        $dEncOutputs = $dOptions['enc_outputs'];
        [$dInputs,$dStates] = $this->encoder->backward($dEncOutputs,$dStates);
        return $dInputs;
    }

    public function predict(NDArray $inputs, array $options=null) : NDArray
    {
        $K = $this->backend;
        $attentionPlot = $options['attention_plot'];

        if($inputs->ndim()!=2) {
            throw new InvalidArgumentException('inputs shape must be 2D.');
        }
        $batchs = $inputs->shape()[0];
        if($batchs!=1) {
            throw new InvalidArgumentException('num of batch must be one.');
        }
        $status = [$K->zeros([$batchs, $this->units])];
        [$encOutputs, $status] = $this->encoder->forward($inputs, $training=false, $status);

        $decInputs = $K->array([[$this->startVocId]],$inputs->dtype());

        $result = [];
        $this->setShapeInspection(false);
        for($t=0;$t<$this->outputLength;$t++) {
            [$predictions, $status] = $this->decoder->forward(
                $decInputs, $training=false, $status, ['enc_outputs'=>$encOutputs]);

            # storing the attention weights to plot later on
            $scores = $this->decoder->getAttentionScores();
            $K->copy($scores->reshape([$this->inputLength]),$attentionPlot[$t]);

            $predictedId = $K->argmax($predictions[0][0]);

            $result[] = $predictedId;

            if($this->endVocId == $predictedId)
                break;

            # the predicted ID is fed back into the model
            $decInputs = $K->array([[$predictedId]],$inputs->dtype());
        }
        $this->setShapeInspection(true);
        $result = $K->array([$result],NDArray::int32);
        #return result, sentence, attention_plot
        return $result;
    }

    public function plotAttention(
        $attention, $sentence, $predictedSentence)
    {
        $plt = $this->plt;
        $plt->figure();
        #attention = attention[:len(predicted_sentence), :len(sentence)]
        $plt->imshow($attention, $cmap='viridis');

        $plt->xticks($this->mo->arange(count($sentence)),$sentence);
        $plt->yticks($this->mo->arange(count($predictedSentence)),$predictedSentence);
    }
}

$numExamples=30000;#30000
$numWords=256;
$epochs = 10;#10
$batchSize = 64;
$wordVectSize=256;#256
$units=256;#1024


$mo = new MatrixOperator();
$nn = new NeuralNetworks($mo);
$pltConfig = [
    'figure.bottomMargin'=>100,
    'frame.xTickLabelAngle'=>90,
];
$plt = new Plot($pltConfig,$mo);

$dataset = new EngFraDataset($mo);

echo "Generating data...\n";
[$inputTensor, $targetTensor, $inpLang, $targLang]
    = $dataset->loadData(null,$numExamples,$numWords);
$valSize = intval(floor(count($inputTensor)/10));
$trainSize = count($inputTensor)-$valSize;
$inputTensorTrain  = $inputTensor[[0,$trainSize-1]];
$targetTensorTrain = $targetTensor[[0,$trainSize-1]];
$inputTensorVal  = $inputTensor[[$trainSize,$valSize+$trainSize-1]];
$targetTensorVal = $targetTensor[[$trainSize,$valSize+$trainSize-1]];

$inputLength  = $inputTensor->shape()[1];
$outputLength = $targetTensor->shape()[1];
$inputVocabSize = $inpLang->numWords();
$targetVocabSize = $targLang->numWords();
$corpusSize = count($inputTensor);

echo "num_examples: $numExamples\n";
echo "num_words: $numWords\n";
echo "epoch: $epochs\n";
echo "embedding_dim: $wordVectSize\n";
echo "units: $units\n";
echo "Total questions: $corpusSize\n";
echo "Input  word dictionary: $inputVocabSize(".$inpLang->numWords(true).")\n";
echo "Target word dictionary: $targetVocabSize(".$targLang->numWords(true).")\n";
echo "Input length: $inputLength\n";
echo "Output length: $outputLength\n";


$seq2seq = new Seq2seq(
    $mo,
    $nn->backend(),
    $nn,
    $inputLength,
    $inputVocabSize,
    $outputLength,
    $targetVocabSize,
    $wordVectSize,
    $units,
    $targLang->wordToIndex('<start>'),
    $targLang->wordToIndex('<end>'),
    $plt
);

echo "Compile model...\n";
$seq2seq->compile([
    'loss'=>'sparse_categorical_crossentropy',
    'optimizer'=>'adam',
    'metrics'=>['accuracy','loss'],
]);

#$a=$inpLang->sequencesToTexts($inputTensorTrain[[0,10]]);
#$v=$targLang->sequencesToTexts($targetTensorTrain[[0,10]]);
#foreach(array_map(null,$a->getArrayCopy(),$v->getArrayCopy()) as  $values) {
#    [$i,$t] = $values;
#    echo "input:".$i."\n";
#    echo "target:".$t."\n";
#}
#echo "---------\n";
#$a=$inpLang->sequencesToTexts($inputTensorVal[[0,10]]);
#$v=$targLang->sequencesToTexts($targetTensorVal[[0,10]]);
#foreach(array_map(null,$a->getArrayCopy(),$v->getArrayCopy()) as  $values) {
#    [$i,$t] = $values;
#    echo "input: ".$i."\n";
#    echo "target:".$t."\n";
#}
#exit();
echo "Train model...\n";
$history = $seq2seq->fit(
    $inputTensorTrain,
    $targetTensorTrain,
    [
        'batch_size'=>$batchSize,
        'epochs'=>$epochs,
        'validation_data'=>[$inputTensorVal,$targetTensorVal],
        #callbacks=[checkpoint],
    ]);

$choice = $mo->random()->choice($corpusSize,10,false);
foreach($choice as $idx)
{
    $question = $inputTensor[$idx]->reshape([1,$inputLength]);
    $attentionPlot = $mo->zeros([$outputLength, $inputLength]);
    $predict = $seq2seq->predict(
        $question,['attention_plot'=>$attentionPlot]);
    $answer = $targetTensor[$idx]->reshape([1,$outputLength]);;
    $sentence = $inpLang->sequencesToTexts($question)[0];
    $predictedSentence = $targLang->sequencesToTexts($predict)[0];
    $targetSentence = $targLang->sequencesToTexts($answer)[0];
    echo "Input:   $sentence\n";
    echo "Predict: $predictedSentence\n";
    echo "Target:  $targetSentence\n";
    echo "\n";
    #attention_plot = attention_plot[:len(predicted_sentence.split(' ')), :len(sentence.split(' '))]
    $seq2seq->plotAttention($attentionPlot,  explode(' ',$sentence), explode(' ',$predictedSentence));
}
$plt->figure();
$plt->plot($mo->array($history['accuracy']),null,null,'accuracy');
$plt->plot($mo->array($history['val_accuracy']),null,null,'val_accuracy');
$plt->plot($mo->array($history['loss']),null,null,'loss');
$plt->plot($mo->array($history['val_loss']),null,null,'val_loss');
$plt->legend();
$plt->title('seq2seq-attention-translation');
$plt->show();
