<?php
namespace Rindow\NeuralNetworks\Data\Sequence;

use Rindow\NeuralNetworks\Data\Dataset\ClassifiedDirectoryDataset;
use Interop\Polite\Math\Matrix\NDArray;
use ArrayObject;

class TextClassifiedDataset extends ClassifiedDirectoryDataset
{
    protected $verbose;

    public function __construct(
        $mo, string $path, array $options=null
        )
    {
        $leftargs = [];
        parent::__construct($mo,$path,$options,$leftargs);
        if(array_key_exists('verbose',$leftargs)) {
            $this->verbose = $leftargs['verbose'];
            unset($leftargs['verbose']);
        }
        if($this->filter==null) {
            $this->filter = new TextFilter($mo,$leftargs);
        }
    }

    public function getTokenizer()
    {
        return $this->filter->getTokenizer();
    }

    public function getPreprocessor()
    {
        return $this->filter->getPreprocessor();
    }

    public function getLabels()
    {
        return $this->filter->labels();
    }

    protected function console($message)
    {
        if($this->verbose) {
            if(defined('STDERR')) {
                fwrite(STDERR,$message);
            }
        }
    }

    public function fitOnTexts($loadAll=null,$noFit=null)
    {
        $filter = $this->filter;
        $this->filter = null;
        $unclassified = $this->unclassified;
        $this->unclassified = !$loadAll;
        $batchSize = $this->batchSize;
        $this->batchSize = 0;
        $tokenizer = $filter->getTokenizer();
        if($loadAll) {
            $labels = $filter->labels();
            $labelNum = count($labels);
            $inputs = new ArrayObject();
            $tests = new ArrayObject();
            $nn=0;
            $this->console('Indexing ...');
            $dataset = $this->getIterator();
            $this->console(" Done.\n");
            $this->console('Loding ');
            foreach($dataset as $data) {
                $inputs->append($data[0]);
                $label = $data[1];
                $tests->append($label);
                if(!array_key_exists($label,$labels)) {
                    $labels[$label] = $labelNum;
                    $labelNum++;
                }
                if($nn>1000) {
                    $this->console('.');
                    $nn=0;
                }
                $nn++;
            }
            $filter->setLabels($labels);
        } else {
            $inputs = $this->getIterator();
        }
        $this->console(" Done.\n");
        if(!$noFit) {
            $this->console("Fitting ...");
            $tokenizer->fitOnTexts($inputs);
            $this->console(" Done.\n");
        }
        $this->filter = $filter;
        $this->unclassified = $unclassified;
        $this->batchSize = $batchSize;
        if($loadAll) {
            return [$inputs,$tests];
        }
    }

    public function classnames()
    {
        return $this->filter->classnames();
    }

    public function loadData()
    {
        $tokenizer = $this->filter->getTokenizer();
        $noFit = count($tokenizer->wordCounts())>0?true:false;
        [$inputs,$tests] = $this->fitOnTexts($loadAll=true,$noFit);
        $this->console("Generating sequences ...");
        $sequences = $tokenizer->textsToSequences($inputs);
        $this->console(" Done.\n");
        $this->console("Formatting sequences ...");
        $sets = $this->filter->translate($inputs, $tests);
        $this->console(" Done.\n");
        return $sets;
        //$inputs = $this->filter->getPreprocessor()
        //    ->padSequences($sequences,[
        //        'maxlen'=>$this->maxlen,
        //        'dtype'=>$this->dtype,
        //        'padding'=>$this->padding,
        //        'truncating'=>$this->truncating,
        //        'value'=>$this->value,
        //    ]);
        //$this->console(" Done.\n");
        //$ntests = $this->mo->la()->alloc([count($tests)],NDArray::int32);
        //$labels = $this->filter->labels();
        //$this->console("Generating labels ...");
        //foreach($tests as $i => $value) {
        //    $ntests[$i] = $labels[$value];
        //}
        //$this->console(" Done.\n");
        //return [$inputs,$ntests];
    }
}
