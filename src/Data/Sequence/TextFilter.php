<?php
namespace Rindow\NeuralNetworks\Data\Sequence;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Data\Dataset\DatasetFilter;
use InvalidArgumentException;

class TextFilter implements DatasetFilter
{
    protected $mo;
    protected $labels = [];
    protected $labelNum = 0;

    public function __construct(
        object $mo,
        object $tokenizer=null,
        array $classnames=null,
        callable $analyzer=null,
        int $num_words=null,
        string $tokenizer_filters=null,
        string $specials=null,
        bool $lower=null,
        string $split=null,
        bool $char_level=null,
        string $oov_token=null,
        int $document_count=null,
        object $preprocessor=null,
        int $maxlen=null,
        int $dtype=null,
        string $padding=null,
        string $truncating=null,
        float|int|bool $value=null,
    )
    {
        $tokenizer = $tokenizer ?? null;
        $classnames = $classnames ?? [];
        $analyzer = $analyzer ?? null;
        $num_words = $num_words ?? null;
        $tokenizer_filters = $tokenizer_filters ?? "!\"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r";
        $specials = $specials ?? null;
        $lower = $lower ?? true;
        $split = $split ?? " ";
        $char_level = $char_level ?? false;
        $oov_token = $oov_token ?? null;
        $document_count = $document_count ?? 0;
        $preprocessor = $preprocessor ?? null;
        $maxlen = $maxlen ?? null;
        $dtype = $dtype ?? NDArray::int32;
        $padding = $padding ?? "post";
        $truncating = $truncating ?? "post";
        $value = $value ?? 0;

        $this->mo = $mo;
        if($tokenizer==null) {
            $tokenizer = new Tokenizer($mo,
                analyzer: $analyzer,
                num_words: $num_words,
                filters: $tokenizer_filters,
                specials: $specials,
                lower: $lower,
                split: $split,
                char_level: $char_level,
                oov_token: $oov_token,
                document_count: $document_count,
            );
        }
        $this->tokenizer = $tokenizer;
        if($preprocessor==null) {
            $preprocessor = new Preprocessor($mo);
        }
        $this->labels = array_flip($classnames);
        $this->preprocessor = $preprocessor;
        //if($maxlen==null||$maxlen<1) {
        //    throw new InvalidArgumentException('maxlen must be greater then 0');
        //}
        $this->maxlen=$maxlen;
        $this->dtype=$dtype;
        $this->padding=$padding;
        $this->truncating=$truncating;
        $this->value=$value;
    }

    public function getTokenizer()
    {
        return $this->tokenizer;
    }

    public function getPreprocessor()
    {
        return $this->preprocessor;
    }

    public function classnames()
    {
        return array_flip($this->labels);
    }

    public function labels()
    {
        return $this->labels;
    }

    public function setLabels($labels)
    {
        $this->labels = $labels;
    }

    public function classnameToIndex(string $word) : int
    {
        if(!array_key_exists($word,$this->labels))
            return null;
        return $this->labels[$word];
    }

    public function indexToClassname(int $index) : string
    {
        $classname = array_search($index,$this->labels);
        if($classname===false)
            return null;
        return $classname;
    }

    public function translate(
        iterable $inputs,
        iterable $tests=null,
        $options=null) : array
    {
        //$this->tokenizer->fitOnTexts($inputs);
        $sequences = $this->tokenizer->textsToSequences($inputs);
        $inputsArray = $this->preprocessor->padSequences($sequences,
            maxlen: $this->maxlen,
            dtype: $this->dtype,
            padding: $this->padding,
            truncating: $this->truncating,
            value: $this->value,
        );

        $testsCount = count($tests);
        $testsArray = $this->mo->la()->alloc([$testsCount],$this->dtype);
        foreach ($tests as $key => $label) {
            if(array_key_exists($label,$this->labels)) {
                $labelNum = $this->labels[$label];
            } else {
                $this->labels[$label] = $this->labelNum;
                $labelNum = $this->labelNum;
                $this->labelNum++;
            }
            $testsArray[$key] = $labelNum;
        }

        return [$inputsArray,$testsArray];
    }
}
