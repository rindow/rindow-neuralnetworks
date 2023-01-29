<?php
namespace Rindow\NeuralNetworks\Data\Dataset;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\Dir;
use ArrayObject;
use InvalidArgumentException;
use LogicException;
use Countable;
use IteratorAggregate;
use Traversable;

class ClassifiedDirectoryDataset implements Countable,IteratorAggregate,Dataset
{
    protected $mo;
    protected $path;
    protected $pattern;
    protected $batchSize;
    protected $crawler;
    protected $filter;
    protected $unclassified;
    protected $shuffle;
    protected $limit;
    protected $restrictedByClass;
    protected $filenames;
    protected $maxSteps=0;
    protected $maxDatasetSize=0;

    public function __construct(
        object $mo,
        string $path,
        string $pattern=null,
        int $batch_size=null,
        object $crawler=null,
        DatasetFilter $filter=null,
        bool $unclassified=null,
        bool $shuffle=null,
        int $limit=null,
        array $restricted_by_class=null,
    )
    {
        $pattern = $pattern ?? null;
        $batch_size = $batch_size ?? 32;
        $crawler = $crawler ?? null;
        $filter = $filter ?? null;
        $unclassified = $unclassified ?? false;
        $shuffle = $shuffle ?? false;
        $limit = $limit ?? null;
        $restricted_by_class = $restricted_by_class ?? null;

        $this->mo = $mo;
        $this->crawler = $crawler;
        $this->path = $path;
        $this->pattern = $pattern;
        $this->batchSize = $batch_size;
        if($crawler==null) {
            $crawler = new Dir();
        }
        $this->crawler = $crawler;
        $this->filter = $filter;
        $this->unclassified = $unclassified;
        $this->shuffle = $shuffle;
        $this->limit = $limit;
        if($restricted_by_class) {
            $this->restrictedByClass = array_flip($restricted_by_class);
        }
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

    public function count() : int
    {
        return $this->maxSteps;
    }

    protected function getFilenames()
    {
        if($this->filenames!==null) {
            return $this->filenames;
        }
        $filenames = $this->crawler->glob($this->path,$this->pattern);
        $rawfilenames = $filenames;
        $filenames = [];
        $prefixLength = strlen($this->path.DIRECTORY_SEPARATOR);
        foreach ($rawfilenames as $filename) {
            $sepfilename = explode(DIRECTORY_SEPARATOR,substr($filename,$prefixLength));
            $label = $sepfilename[0];
            if(count($sepfilename)<2) {
                continue;
            }
            if($this->restrictedByClass &&
                !array_key_exists($label,$this->restrictedByClass)) {
                continue;
            }
            $filenames[] = $filename;
        }

        $size = count($filenames);
        if($this->shuffle && $size>0) {
            $choice = $this->mo->la()->randomSequence($size);
            $newFilenames = [];
            foreach ($choice as $idx) {
                $newFilenames[] = $filenames[$idx];
            }
            $filenames = $newFilenames;
        }
        if($this->limit) {
            $filenames = array_slice($filenames,0,$this->limit);
        }
        $this->filenames = $filenames;
        return $this->filenames;
    }

    protected function readContents($filename)
    {
        return file_get_contents($filename);
    }

    protected function makeBatchInputs($inputs)
    {
        return $inputs;
    }

    protected function makeBatchTests($tests)
    {
        return $tests;
    }

    public function  getIterator() : Traversable
    {
        $la = $this->mo->la();
        $filenames = $this->getFilenames();
        $prefixLength = strlen($this->path.DIRECTORY_SEPARATOR);
        $this->maxDatasetSize = 0;
        $rows = 0;
        $steps = 0;
        $inputs = [];
        $tests = [];
        $paths = [];
        foreach($filenames as $filename) {
            $sepfilename = explode(DIRECTORY_SEPARATOR,substr($filename,$prefixLength));
            $label = $sepfilename[0];
            if(count($sepfilename)<2) {
                continue;
            }
            $content = $this->readContents($filename);
            if($this->batchSize==0) {
                // stream mode
                if($this->unclassified) {
                    $data = $content;
                } else {
                    $data = [$content,$label];
                }
                yield $rows => $data;
                $rows++;
                continue;
            }
            $inputs[] = $content;
            $tests[] = $label;
            $paths[] = $filename;
            $rows++;
            if($rows>=$this->batchSize) {
                $inputs = $this->makeBatchInputs($inputs);
                $tests = $this->makeBatchTests($tests);
                $inputsets = [$inputs,$tests];
                if($this->filter) {
                    $inputsets = $this->filter->translate($inputs,$tests,$paths);
                }
                $this->maxDatasetSize += $rows;
                $rows = 0;
                if($this->unclassified) {
                    $data = $inputsets[0];
                } else {
                    $data = $inputsets;
                }
                yield $steps => $data;
                $steps++;
                $this->maxSteps = max($this->maxSteps,$steps);
                $inputs = [];
                $tests = [];
                $paths = [];
            }
        }
        $this->maxDatasetSize += $rows;
        if($this->batchSize==0) {
            // stream mode
            return;
        }
        if($rows) {
            $inputs = $this->makeBatchInputs($inputs);
            $tests = $this->makeBatchTests($tests);
            $inputsets = [$inputs,$tests];
            if($this->filter) {
                $inputsets = $this->filter->translate($inputs,$tests,$paths);
            }
            if($this->unclassified) {
                $data = $inputsets[0];
            } else {
                $data = $inputsets;
            }
            yield $steps => $data;
            $steps++;
            $this->maxSteps = max($this->maxSteps,$steps);
        }
    }

    protected function console($message)
    {

    }

    protected function progressBar($done,$total,$startTime,$maxDot)
    {
        if($done==0) {
            $this->console("\r{$done}/{$total} ");
            return;
        }
        $elapsed = time() - $startTime;
        if($total) {
            $completion = $done/$total;
            $estimated = $elapsed / $completion;
            $remaining = $estimated - $elapsed;
            $dot = (int)ceil($maxDot*$completion);
            $sec = $remaining % 60;
            $min = (int)floor($remaining/60) % 60;
            $hour = (int)floor($remaining/3600);
            $rem_string = ($hour?$hour.':':'').sprintf('%02d:%02d',$min,$sec);
        } else {
            $dot = 1;
            $rem_string = '????';
            $this->console($maxDot."\n");
        }
        $this->console("\r{$done}/{$total} [".str_repeat('.',$dot).str_repeat(' ',$maxDot-$dot).
            "] {$elapsed} sec. remaining:{$rem_string}  ");
    }
}
