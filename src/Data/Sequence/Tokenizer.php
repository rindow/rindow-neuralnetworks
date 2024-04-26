<?php
namespace Rindow\NeuralNetworks\Data\Sequence;

use InvalidArgumentException;
use ArrayObject;

class Tokenizer
{
    protected object $mo;
    protected mixed $analyzer;
    protected ?int $numWords;
    protected ?string $filters;
    protected ?string $specials;
    protected bool $lower;
    protected string $split;
    protected bool $charLevel;
    protected ?string $oovToken;
    protected int $documentCount;
    /** @var array<string,int> $wordCounts */
    protected array $wordCounts = [];
    /** @var array<string,int> $wordToIndex */
    protected array $wordToIndex = [];
    /** @var array<int,string> $indexToWord */
    protected array $indexToWord = [];
    /** @var array<string,int> $filtersMap */
    protected ?array $filtersMap = null;
    /** @var array<string,int> $specialsMap */
    protected ?array $specialsMap = null;

    public function __construct(
        object $mo,
        callable $analyzer=null,
        int $num_words=null,
        string $filters=null,
        string $specials=null,
        bool $lower=null,
        string $split=null,
        bool $char_level=null,
        string $oov_token=null,
        int $document_count=null,
    )
    {
        $this->mo = $mo;
        $analyzer = $analyzer ?? null;
        $num_words = $num_words ?? null;
        $filters = $filters ?? "!\"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n";
        $specials = $specials ?? null;
        $lower = $lower ?? true;
        $split = $split ?? " ";
        $char_level = $char_level ?? false;
        $oov_token = $oov_token ?? null;
        $document_count = $document_count ?? 0;

        $this->analyzer = $analyzer;
        $this->numWords = $num_words;
        $this->filters = $filters;
        $this->specials = $specials;
        $this->lower = $lower;
        $this->split = $split;
        $this->charLevel = $char_level;
        $this->oovToken = $oov_token;
        $this->documentCount = $document_count;
    }

    /**
     * @return array<string>
     */
    protected function textToWordSequence(
        string $text,
        ?string $filters,
        ?string $specials,
        bool $lower,
        string $split
        ) : array
    {
        if($lower)
            $text = mb_strtolower($text);
        $text = mb_str_split($text);
        if($this->filtersMap) {
            $map = $this->filtersMap;
        } else {
            if($filters) {
                $map = $this->filtersMap = array_flip(str_split($filters));
            } else {
                $map = [];
            }
        }
        if($this->specialsMap) {
            $specialsMap = $this->specialsMap;
        } else {
            if($specials) {
                $specialsMap = $this->specialsMap = array_flip(str_split($specials));
            } else {
                $specialsMap = [];
            }
        }
        $func = function($c) use ($map,$specialsMap,$split) {
            if($specialsMap && array_key_exists($c,$specialsMap)) {
                $c = $split.$c.$split;
            } else {
                $c = array_key_exists($c,$map) ? $split : $c;
            }
            return $c;
        };
        $text = implode('',array_map($func,$text));
        $seq = array_values(array_filter(explode(' ',$text)));
        return $seq;
    }

    /**
     * @param iterable<string> $texts
     */
    public function fitOnTexts(iterable $texts) : void
    {
        foreach ($texts as $text) {
            $this->documentCount++;
            if($this->analyzer) {
                $analyzer = $this->analyzer;
                $seq = $analyzer($text);
            } else {
                $seq = $this->textToWordSequence(
                    $text,$this->filters,$this->specials,$this->lower,$this->split);
            }
            foreach ($seq as $w) {
                if(is_float($w)) {
                    $w = (int)$w;
                }
                if(array_key_exists($w,$this->wordCounts)) {
                    $this->wordCounts[$w]++;
                } else {
                    $this->wordCounts[$w] = 1;
                }
            }
        }
        arsort($this->wordCounts);
        if($this->oovToken) {
            $this->wordCounts = array_merge([$this->oovToken=>1],$this->wordCounts);
        }
        $idx = 1;
        foreach ($this->wordCounts as $key => $value) {
            $this->wordToIndex[$key] = $idx;
            $this->indexToWord[$idx] = $key;
            $idx++;
        }
    }
    //public function fitOnSequences($sequences) : void
    //{
    //}

    /**
     * @param iterable<string> $texts
     * @return iterable<iterable<int>>
     */
    public function textsToSequences(iterable $texts) : iterable
    {
        $sequences = new ArrayObject();
        foreach($this->textsToSequencesGenerator($texts) as $seq) {
            $sequences->append($seq);
        }
        return $sequences;
    }

    /**
     * @param iterable<string> $texts
     * @return iterable<iterable<int>>
     */
    public function textsToSequencesGenerator(iterable $texts) : iterable
    {
        $numWords = $this->numWords;
        $oovTokenIndex = ($this->oovToken) ?
            $this->wordToIndex[$this->oovToken] : null;
        foreach ($texts as $text) {
            if($this->analyzer) {
                $analyzer = $this->analyzer;
                $seq = $analyzer($text);
            } else {
                $seq = $this->textToWordSequence(
                    $text,$this->filters,$this->specials,$this->lower,$this->split);
            }
            $vect = [];
            foreach ($seq as $w) {
                if(array_key_exists($w,$this->wordToIndex)) {
                    $i = $this->wordToIndex[$w];
                    if($numWords==null || $i<$numWords) {
                        $vect[] = $i;
                    } else {
                        if($oovTokenIndex) {
                            $vect[] = $oovTokenIndex;
                        }
                    }
                } else {
                    if($oovTokenIndex) {
                        $vect[] = $oovTokenIndex;
                    }
                }
            }
            yield $vect;
        }
    }

    /**
     * @param iterable<iterable<int>> $sequences
     * @return iterable<string>
     */
    public function sequencesToTexts(iterable $sequences) : iterable
    {
        $texts = new ArrayObject();
        foreach($this->sequencesToTextsGenerator($sequences) as $text) {
            $texts->append($text);
        }
        return $texts;
    }

    /**
     * @param iterable<iterable<int>> $sequences
     * @return iterable<string>
     */
    public function sequencesToTextsGenerator(iterable $sequences) : iterable
    {
        if(!is_iterable($sequences)) {
            throw new InvalidArgumentException('sequences must be list of sequence.');
        }
        $numWords = $this->numWords;
        $oovTokenIndex = ($this->oovToken) ?
            $this->wordToIndex[$this->oovToken] : null;
        foreach ($sequences as $seq) {
            if(!is_iterable($seq)) {
                throw new InvalidArgumentException('sequences must be list of sequence.');
            }
            $vect = [];
            foreach ($seq as $num) {
                if(!is_int($num)) {
                    throw new InvalidArgumentException('sequence includes "'.gettype($num).'". it must be numeric.');
                }
                if(array_key_exists($num,$this->indexToWord)) {
                    $w = $this->indexToWord[$num];
                    if($numWords==null || $num<$numWords) {
                        $vect[] = $w;
                    } else {
                        if($oovTokenIndex) {
                            $vect[] = $this->oovToken;
                        }
                    }
                } else {
                    if($oovTokenIndex) {
                        $vect[] = $this->oovToken;
                    }
                }
            }
            $vect = implode($this->split,$vect);
            yield $vect;
        }
    }

    public function documentCount() : int
    {
        return $this->documentCount;
    }

    /**
     * @return array<string,int>
     */
    public function wordCounts() : array
    {
        return $this->wordCounts;
    }

    /**
     * @return array<int,string>
     */
    public function getWords() : array
    {
        return $this->indexToWord;
    }

    public function numWords(bool $internal=null) : int
    {
        if($internal||$this->numWords===null) {
            return count($this->indexToWord)+1;
        } else {
            return min(count($this->indexToWord)+1,$this->numWords);
        }
    }

    public function wordToIndex(string $word) : int
    {
        if(!array_key_exists($word,$this->wordToIndex)) {
            throw new InvalidArgumentException("No matching word found");
        }
        return $this->wordToIndex[$word];
    }

    public function indexToWord(int $index) : string
    {
        if(!array_key_exists($index,$this->indexToWord)) {
            throw new InvalidArgumentException("No matching word found");
        }
        return $this->indexToWord[$index];
    }

    public function save() : string
    {
        return serialize([
            $this->documentCount,
            $this->wordCounts,
            $this->wordToIndex,
            $this->indexToWord,
        ]);
    }

    public function load(string $string) : void
    {
        [
            $this->documentCount,
            $this->wordCounts,
            $this->wordToIndex,
            $this->indexToWord,
        ] = unserialize($string);
    }
}
