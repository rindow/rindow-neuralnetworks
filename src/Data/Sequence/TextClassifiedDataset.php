<?php
namespace Rindow\NeuralNetworks\Data\Sequence;

use Rindow\NeuralNetworks\Data\Dataset\ClassifiedDirectoryDataset;
use Interop\Polite\Math\Matrix\NDArray;
use ArrayObject;

/**
 * @extends ClassifiedDirectoryDataset<string>
 */
class TextClassifiedDataset extends ClassifiedDirectoryDataset
{
    protected ?int $verbose;
    protected ?object $filter;

    public function __construct(
        object $mo,
        string $path,
        ?int $verbose=null,
        mixed ...$options
        )
    {
        $opts = [];
        foreach([
            'pattern', 'batch_size', 'crawler', 'filter', 
            'unclassified', 'shuffle', 'limit','restricted_by_class',
            ] as $o) {
            if(isset($options[$o])) {
                $opts[$o] = $options[$o];
                unset($options[$o]);
            }
        }
        parent::__construct($mo,$path, ...$opts);
        $this->verbose = $verbose;
        if($this->filter()==null) {
            $this->filter = new TextFilter($mo, ...$options);
            $this->setFilter($this->filter);
        }
    }

    public function getTokenizer() : object
    {
        return $this->filter->getTokenizer();
    }

    public function getPreprocessor() : object
    {
        return $this->filter->getPreprocessor();
    }

    /**
     * @return array<string,int>
     */
    public function getLabels() : array
    {
        return $this->filter->labels();
    }

    protected function console(string $message) : void
    {
        if($this->verbose) {
            if(defined('STDERR')) {
                fwrite(STDERR,$message);
            }
        }
    }

    /**
     * @return array{iterable<string>,iterable<string>}
     */
    public function fitOnTexts(
        ?bool $loadAll=null, ?bool $noFit=null) : ?array
    {
        $filter = $this->filter;
        $this->filter = null;
        $this->setFilter(null);
        
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
            $filenames = $this->getFilenames();
            $totalSize = count($filenames);
            $this->console(" Done. Total=$totalSize\n");
            $this->console("Loding ...\n");
            /** @var iterable<string>  $dataset */
            $dataset = $this->getIterator();
            $startTime = time();
            $count = 0;
            foreach($dataset as $data) {
                $inputs->append($data[0]);
                $label = $data[1];
                $tests->append($label);
                if(!array_key_exists($label,$labels)) {
                    $labels[$label] = $labelNum;
                    $labelNum++;
                }
                $count++;
                $nn++;
                if($nn>=1000) {
                    $this->progressBar($count,$totalSize,$startTime,25);
                    $nn=0;
                }
            }
            $this->progressBar($count,$totalSize,$startTime,25);
            $filter->setLabels($labels);
        } else {
            /** @var iterable<string>  $inputs */
            $inputs = $this->getIterator();
        }
        $this->console("\nDone.\n");
        if(!$noFit) {
            $this->console("Fitting ...");
            $tokenizer->fitOnTexts($inputs);
            $this->console(" Done.\n");
        }
        $this->filter = $filter;
        $this->setFilter($filter);
        $this->unclassified = $unclassified;
        $this->batchSize = $batchSize;
        if($loadAll) {
            return [$inputs,$tests];
        }
        return null;
    }

    /**
     * @return array<string>
     */
    public function classnames() : array
    {
        return $this->filter->classnames();
    }

    /**
     * @return array{NDArray,NDArray}  {inputsArray,testsArray}
     */
    public function loadData() : array
    {
        $tokenizer = $this->filter->getTokenizer();
        $noFit = count($tokenizer->wordCounts())>0?true:false;
        [$inputs,$tests] = $this->fitOnTexts(loadAll:true,noFit:$noFit);
        $this->console("Generating sequences ...");
        $sequences = $tokenizer->textsToSequences($inputs);
        $this->console(" Done.\n");
        $this->console("Formatting sequences ...");
        $sets = $this->filter->translate($inputs, $tests);
        $this->console(" Done.\n");
        return $sets;
    }
}
