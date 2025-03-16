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

/**
 * @template T
 * @implements Dataset<T>
 * @implements IteratorAggregate<int,array{NDArray|array<NDArray>,NDArray}>
 */
class ClassifiedDirectoryDataset implements IteratorAggregate,Dataset
{
    protected object $mo;
    protected string $path;
    protected ?string $pattern;
    protected int $batchSize;
    protected object $crawler;
    /** @var DatasetFilter<T> */
    protected ?DatasetFilter $filterGeneric;
    protected bool $unclassified;
    protected bool $shuffle;
    protected ?int $limit;
    /** @var array<string,int> $restrictedByClass */
    protected ?array $restrictedByClass=null;
    /** @var array<string> $filenames */
    protected ?array $filenames=null;
    protected int $maxSteps=0;
    protected int $maxDatasetSize=0;

    /**
     * @param array<string> $restricted_by_class
     * @param DatasetFilter<T> $filter
     */
    public function __construct(
        object $mo,
        string $path,
        ?string $pattern=null,
        ?int $batch_size=null,
        ?object $crawler=null,
        ?DatasetFilter $filter=null,
        ?bool $unclassified=null,
        ?bool $shuffle=null,
        ?int $limit=null,
        ?array $restricted_by_class=null,
    )
    {
        $pattern = $pattern ?? null;
        $batch_size = $batch_size ?? 32;
        if($crawler==null) {
            $crawler = new Dir();
        }
        $filter = $filter ?? null;
        $unclassified = $unclassified ?? false;
        $shuffle = $shuffle ?? false;
        $limit = $limit ?? null;
        $restricted_by_class = $restricted_by_class ?? null;

        $this->mo = $mo;
        $this->path = $path;
        $this->pattern = $pattern;
        $this->batchSize = $batch_size;
        $this->crawler = $crawler;
        $this->setFilter($filter);
        $this->unclassified = $unclassified;
        $this->shuffle = $shuffle;
        $this->limit = $limit;
        if($restricted_by_class) {
            $this->restrictedByClass = array_flip($restricted_by_class);
        }
    }

    public function filter() : ?DatasetFilter
    {
        return $this->filterGeneric;
    }

    public function setFilter(?DatasetFilter $filter) : void
    {
        $this->filterGeneric = $filter;
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

    /**
     * @return array<string>
     */
    protected function getFilenames() : array
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

    protected function readContents(string $filename) : mixed
    {
        return file_get_contents($filename);
    }

    /**
     * @param iterable<int,NDArray> $inputs
     */
    protected function makeBatchInputs(mixed $inputs) : mixed
    {
        return $inputs;
    }

    /**
     * @param iterable<int,string> $tests
     */
    protected function makeBatchTests(mixed $tests) : mixed
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
                if($this->filter()) {
                    $inputsets = $this->filter()->translate($inputs,$tests,$paths);
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
            if($this->filter()) {
                $inputsets = $this->filter()->translate($inputs,$tests,$paths);
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

    protected function console(string $message) : void
    {

    }

    protected function progressBar(int $done,int $total,int $startTime,int $maxDot) : void
    {
        if($done==0) {
            $this->console("\r{$done}/{$total} ");
            return;
        }
        $elapsed = time() - $startTime;
        if($total) {
            $completion = $done/$total;
            $estimated = (int)floor($elapsed / $completion);
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
