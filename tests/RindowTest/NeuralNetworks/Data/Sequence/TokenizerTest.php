<?php
namespace RindowTest\NeuralNetworks\Data\Sequence\TokenizerTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Data\Sequence\Tokenizer;

class Test extends TestCase
{
    public function testNormal()
    {
        $mo = new MatrixOperator();
        $tok = new Tokenizer($mo);
        $x = [
            "Hello.\n",
            "I am Tom.\n",
            "How are you?\n",
            "Hello Tom.\n",
            "I am fine.\n",
            "I am Jerry.\n",
            "How are you?\n",
            "I am fine too.\n",
        ];

        $tok->fitOnTexts($x);
        $this->assertEquals(11,$tok->numWords());
        $this->assertTrue(is_int($tok->wordToIndex('tom')));
        $this->assertEquals('tom',$tok->indexToWord($tok->wordToIndex('tom')));

        $seq = $tok->textsToSequences($x);
        $this->assertCount(8,$seq);

        $this->assertCount(1,$seq[0]);
        $this->assertCount(3,$seq[1]);
        $this->assertCount(3,$seq[2]);
        $this->assertCount(2,$seq[3]);
        $this->assertCount(3,$seq[4]);
        $this->assertCount(3,$seq[5]);
        $this->assertCount(3,$seq[6]);
        $this->assertCount(4,$seq[7]);

        $this->assertTrue(is_int($seq[0][0]));

        $newTexts = $tok->sequencesToTexts($seq);
        $this->assertCount(8,$newTexts);
        $this->assertEquals([
            "hello",
            "i am tom",
            "how are you",
            "hello tom",
            "i am fine",
            "i am jerry",
            "how are you",
            "i am fine too",
        ],$newTexts->getArrayCopy());
    }

    public function testWithNumWords()
    {
        $mo = new MatrixOperator();
        $tok = new Tokenizer($mo, num_words: 8);
        $x = [
            "Hello.\n",
            "I am Tom.\n",
            "How are you?\n",
            "Hello Tom.\n",
            "I am fine.\n",
            "I am Jerry.\n",
            "How are you?\n",
            "I am fine too.\n",
        ];

        $tok->fitOnTexts($x);

        $this->assertEquals(8,$tok->numWords());
        $this->assertTrue(is_int($tok->wordToIndex('tom')));
        $this->assertEquals('tom',$tok->indexToWord($tok->wordToIndex('tom')));

        $seq = $tok->textsToSequences($x);
        $this->assertCount(8,$seq);

        #$this->assertCount(1,$seq[0]);
        #$this->assertCount(3,$seq[1]);
        #$this->assertCount(3,$seq[2]);
        #$this->assertCount(2,$seq[3]);
        #$this->assertCount(3,$seq[4]);
        #$this->assertCount(3,$seq[5]);

        $this->assertTrue(is_int($seq[0][0]));

        $newTexts = $tok->sequencesToTexts($seq);
        $this->assertCount(8,$newTexts);
        $this->assertEquals([
            "hello",
            "i am tom",
            "how are you",
            "hello tom",
            "i am",
            "i am",
            "how are you",
            "i am",
            ],$newTexts->getArrayCopy());
    }

    public function testWithFilter()
    {
        $mo = new MatrixOperator();
        $tok = new Tokenizer($mo, filters: "\n");
        $x = [
            "Hello.\n",
            "I am Tom.\n",
            "How are you?\n",
            "Hello Tom.\n",
            "I am fine.\n",
            "I am Jerry.\n",
            "How are you?\n",
            "I am fine too.\n",
        ];

        $tok->fitOnTexts($x);
        $this->assertEquals(13,$tok->numWords());
        $this->assertTrue(is_int($tok->wordToIndex('tom.')));
        $this->assertEquals('tom.',$tok->indexToWord($tok->wordToIndex('tom.')));

        $seq = $tok->textsToSequences($x);
        $this->assertCount(8,$seq);

        $this->assertCount(1,$seq[0]);
        $this->assertCount(3,$seq[1]);
        $this->assertCount(3,$seq[2]);
        $this->assertCount(2,$seq[3]);
        $this->assertCount(3,$seq[4]);
        $this->assertCount(3,$seq[5]);
        $this->assertCount(3,$seq[6]);
        $this->assertCount(4,$seq[7]);

        $this->assertTrue(is_int($seq[0][0]));

        $newTexts = $tok->sequencesToTexts($seq);
        $this->assertCount(8,$newTexts);
        $this->assertEquals([
            "hello.",
            "i am tom.",
            "how are you?",
            "hello tom.",
            "i am fine.",
            "i am jerry.",
            "how are you?",
            "i am fine too.",
        ],$newTexts->getArrayCopy());
    }

    public function testWithSpecials()
    {
        $mo = new MatrixOperator();
        $tok = new Tokenizer($mo, specials: "?.");
        $x = [
            "Hello.\n",
            "I am Tom.\n",
            "How are you?\n",
            "Hello Tom.\n",
            "I am fine.\n",
            "I am Jerry.\n",
            "How are you?\n",
            "I am fine too.\n",
        ];

        $tok->fitOnTexts($x);
        $this->assertEquals(13,$tok->numWords());
        $this->assertTrue(is_int($tok->wordToIndex('tom')));
        $this->assertEquals('tom',$tok->indexToWord($tok->wordToIndex('tom')));

        $seq = $tok->textsToSequences($x);
        $this->assertCount(8,$seq);

        $this->assertCount(2,$seq[0]);
        $this->assertCount(4,$seq[1]);
        $this->assertCount(4,$seq[2]);
        $this->assertCount(3,$seq[3]);
        $this->assertCount(4,$seq[4]);
        $this->assertCount(4,$seq[5]);
        $this->assertCount(4,$seq[6]);
        $this->assertCount(5,$seq[7]);

        $this->assertTrue(is_int($seq[0][0]));

        $newTexts = $tok->sequencesToTexts($seq);
        $this->assertCount(8,$newTexts);
        $this->assertEquals([
            "hello .",
            "i am tom .",
            "how are you ?",
            "hello tom .",
            "i am fine .",
            "i am jerry .",
            "how are you ?",
            "i am fine too .",
        ],$newTexts->getArrayCopy());
    }

    public function testSaveAndLoad()
    {
        $mo = new MatrixOperator();
        $tok = new Tokenizer($mo);
        $x = [
            "Hello.\n",
            "I am Tom.\n",
            "How are you?\n",
            "Hello Tom.\n",
            "I am fine.\n",
            "I am Jerry.\n",
            "How are you?\n",
            "I am fine too.\n",
        ];

        $savedata = $tok->save();
        $tok = new Tokenizer($mo);
        $tok->load($savedata);

        $tok->fitOnTexts($x);
        $this->assertEquals(11,$tok->numWords());
        $this->assertTrue(is_int($tok->wordToIndex('tom')));
        $this->assertEquals('tom',$tok->indexToWord($tok->wordToIndex('tom')));

        $seq = $tok->textsToSequences($x);
        $this->assertCount(8,$seq);

        $this->assertCount(1,$seq[0]);
        $this->assertCount(3,$seq[1]);
        $this->assertCount(3,$seq[2]);
        $this->assertCount(2,$seq[3]);
        $this->assertCount(3,$seq[4]);
        $this->assertCount(3,$seq[5]);
        $this->assertCount(3,$seq[6]);
        $this->assertCount(4,$seq[7]);

        $this->assertTrue(is_int($seq[0][0]));

        $newTexts = $tok->sequencesToTexts($seq);
        $this->assertCount(8,$newTexts);
        $this->assertEquals([
            "hello",
            "i am tom",
            "how are you",
            "hello tom",
            "i am fine",
            "i am jerry",
            "how are you",
            "i am fine too",
        ],$newTexts->getArrayCopy());
    }
}
