<?php
namespace Rindow\NeuralNetworks\Data\Dataset;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;
use Rindow\NeuralNetworks\Support\Dir;
use InvalidArgumentException;
use LogicException;
use Countable;
use IteratorAggregate;

class CSVDataset implements Countable,IteratorAggregate,Dataset
{
    use GenericUtils;
    protected $mo;
    protected $path;
    protected $pattern;
    protected $batchSize;
    protected $skipHeader;
    protected $filter;
    protected $crawler;
    protected $shuffle;
    protected $length;
    protected $delimiter;
    protected $enclosure;
    protected $escape;
    protected $filenames;
    protected $maxSteps=0;
    protected $maxDatasetSize=0;

    public function __construct(
        $mo, string $path, array $options=null,
        array &$leftargs=null
        )
    {
        extract($this->extractArgs([
            'pattern'=>null,
            'batch_size'=>32,
            'skip_header'=>false,
            'filter'=>null,
            'crawler'=>null,
            'shuffle'=>false,
            'length'=>0,
            'delimiter'=>',',
            'enclosure'=>'"',
            'escape'=>'\\',
        ],$options,$leftargs));
        $this->mo = $mo;
        $this->crawler = $crawler;
        $this->path = $path;
        $this->pattern = $pattern;
        $this->batchSize = $batch_size;
        $this->skipHeader = $skip_header;
        $this->filter = $filter;
        if($crawler==null) {
            $crawler = new Dir();
        }
        $this->crawler = $crawler;
        $this->shuffle = $shuffle;
        $this->length = $length;
        $this->delimiter = $delimiter;
        $this->enclosure = $enclosure;
        $this->escape = $escape;
    }

    public function setFilter(DatasetFilter $filter) : void
    {
        $this->filter = $filter;
    }

    public function batchSize() : int
    {
        return $this->batchSize;
    }

    public function datasetSize() : int
    {
        return $this->maxDatasetSize;
    }

    public function count()
    {
        return $this->maxSteps;
    }

    protected function getFilenames()
    {
        if($this->filenames===null) {
            $this->filenames = $this->crawler->glob($this->path,$this->pattern);
        }
        return $this->filenames;
    }

    protected function shuffleData($inputs,$tests)
    {
        $la = $this->mo->la();
        $size = count($inputs);
        if($size>1) {
            $choiceItem = $la->randomSequence($size);
        } else {
            $choiceItem = $la->alloc([1],NDArray::int32);
            $la->zeros($choiceItem);
        }
        $inputs = $la->gather($inputs,$choiceItem);
        if($tests!==null) {
            $tests  = $la->gather($tests,$choiceItem);
        }
        return [$inputs,$tests];
    }

    public function  getIterator()
    {
        if(!($this->filter instanceof DatasetFilter)) {
            throw new LogicException('DatasetFilter is not specified');
        }
        $la = $this->mo->la();
        $filenames = $this->getFilenames();
        $rows = 0;
        $steps = 0;
        $batchSize = $this->batchSize;
        foreach($filenames as $filename) {
            $f = fopen($filename,'r');
            if($this->skipHeader) {
                $row = fgetcsv($f,$this->length,
                            $this->delimiter,$this->enclosure,$this->escape);
                if(!$row) {
                    fclose($f);
                    break;
                }
            }
            while($row = fgetcsv($f,$this->length,
                        $this->delimiter,$this->enclosure,$this->escape)) {
                $inputs[] = $row;
                $rows++;
                if($rows>=$batchSize) {
                    $inputset = $this->filter->translate($inputs);
                    $this->maxDatasetSize += $rows;
                    $rows = 0;
                    if($this->shuffle) {
                        $inputset = $this->shuffleData($inputset[0],$inputset[1]);
                    }
                    yield $steps => $inputset;
                    $steps++;
                    $this->maxSteps = max($this->maxSteps,$steps);
                    $inputs = [];
                    $tests = null;
                }
            }
            fclose($f);
        }
        $this->maxDatasetSize += $rows;
        if($rows) {
            $inputset = $this->filter->translate($inputs);
            if($this->shuffle) {
                $inputset = $this->shuffleData($inputset[0],$inputset[1]);
            }
            yield $steps => $inputset;
            $steps++;
            $this->maxSteps = max($this->maxSteps,$steps);
        }
    }
}
