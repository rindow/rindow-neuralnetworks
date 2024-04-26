<?php
namespace Rindow\NeuralNetworks\Data\Dataset;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\Dir;
use InvalidArgumentException;
use LogicException;
use Countable;
use IteratorAggregate;
use Traversable;

/**
 * @implements Dataset<array<mixed>>
 * @implements IteratorAggregate<int,array{NDArray|array<NDArray>,NDArray}>
 */
class CSVDataset implements IteratorAggregate,Dataset
{
    protected object $mo;
    protected string $path;
    protected ?string $pattern;
    protected int $batchSize;
    protected bool $skipHeader;
    /** @var DatasetFilter<array<mixed>> $filter */
    protected ?DatasetFilter $filter;
    protected object $crawler;
    protected bool $shuffle;
    protected int $length;
    protected string $delimiter;
    protected string $enclosure;
    protected string $escape;
    /** @var array<string> $filenames */
    protected ?array $filenames=null;
    protected int $maxSteps=0;
    protected int $maxDatasetSize=0;

    /**
     * @param DatasetFilter<array<mixed>> $filter
     */
    public function __construct(
        object $mo,
        string $path,
        string $pattern=null,
        int $batch_size=null,
        bool $skip_header=null,
        DatasetFilter $filter=null,
        object $crawler=null,
        bool $shuffle=null,
        int $length=null,
        string $delimiter=null,
        string $enclosure=null,
        string $escape=null,
    )
    {
        // defaults 
        $pattern = $pattern ?? null;
        $batch_size = $batch_size ?? 32;
        $skip_header = $skip_header ?? false;
        $filter = $filter ?? null;
        if($crawler==null) {
            $crawler = new Dir();
        }
        $shuffle = $shuffle ?? false;
        $length = $length ?? 0;
        $delimiter = $delimiter ?? ',';
        $enclosure = $enclosure ?? '"';
        $escape = $escape ?? '\\';

        $this->mo = $mo;
        $this->crawler = $crawler;
        $this->path = $path;
        $this->pattern = $pattern;
        $this->batchSize = $batch_size;
        $this->skipHeader = $skip_header;
        $this->filter = $filter;
        $this->crawler = $crawler;
        $this->shuffle = $shuffle;
        $this->length = $length;
        $this->delimiter = $delimiter;
        $this->enclosure = $enclosure;
        $this->escape = $escape;
    }

    public function filter() : ?DatasetFilter
    {
        return $this->filter;
    }

    public function setFilter(?DatasetFilter $filter) : void
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

    /**
     * @return array<string>
     */
    protected function getFilenames() : array
    {
        if($this->filenames===null) {
            $this->filenames = $this->crawler->glob($this->path,$this->pattern);
        }
        return $this->filenames;
    }

    /**
     * @return array{NDArray,NDArray}
     */
    protected function shuffleData(NDArray $inputs, ?NDArray $tests) : array
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

    public function  getIterator() : Traversable
    {
        if(!($this->filter instanceof DatasetFilter)) {
            throw new LogicException('DatasetFilter is not specified');
        }
        $la = $this->mo->la();
        $filenames = $this->getFilenames();
        $rows = 0;
        $steps = 0;
        $batchSize = $this->batchSize;
        $inputs = [];
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
                    /** @var array{NDArray,NDArray} $inputset */
                    $inputset = $this->filter()->translate($inputs);
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
            /** @var array{NDArray,NDArray} $inputset */
            $inputset = $this->filter()->translate($inputs);
            if($this->shuffle) {
                $inputset = $this->shuffleData($inputset[0],$inputset[1]);
            }
            yield $steps => $inputset;
            $steps++;
            $this->maxSteps = max($this->maxSteps,$steps);
        }
    }
}
