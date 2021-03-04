<?php
namespace RindowTest\NeuralNetworks\Data\Sequence\TextClassifiedDatasetTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Data\Sequence\TextClassifiedDataset;

class Test extends TestCase
{
    public function sortResult($inputs,$tests)
    {
        $results = [];
        $testResults = [];
        foreach ($inputs as $key => $txt) {
            $results['i'.$key] = $txt;
        }
        foreach ($tests as $key => $value) {
            $testResults['i'.$key] = $value;
        }
        asort($results,SORT_STRING);
        $testResults2 = [];
        foreach ($results as $key => $txt) {
            $testResults2[] = $testResults[$key];
        }
        $results = array_values($results);
        return [$results,$testResults2];
    }

    public function int2label($tests,$classnames)
    {
        $testResults = [];
        foreach ($tests as $value) {
            $testResults[] = $classnames[$value];
        }
        return $testResults;
    }

    public function testNormal()
    {
        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);

        $dataset = $nn->data()->TextClassifiedDataset(
            __DIR__.'/../Dataset/text',
            [
                'pattern'=>'@.*\\.txt@',
                'batch_size'=>2,
            ]
        );
        $dataset->fitOnTexts();
        $this->assertEquals(5,$dataset->datasetSize());

        // sequential access
        $datas = [];
        $sets = [];
        $liseqs = [];
        $lilabels = [];
        foreach ($dataset as $key => $value) {
            $datas[] = $value;
            [$texts,$labels] = $value;
            foreach($texts as $key => $text) {
                $label = $labels[$key];
                $sets[] = [$text,$label];
                $liseqs[] = $text;
                $lilabels[] = $label;
            }
        }
        $this->assertCount(3,$datas);
        $this->assertCount(5,$sets);
        $this->assertEquals(3,count($dataset));
        $this->assertEquals(5,$dataset->datasetSize());
        $this->assertInstanceof(NDArray::class,$datas[0][0]);
        //$this->assertEquals([2,3],$datas[0][0]->shape());

        $tokenizer = $dataset->getTokenizer();
        $this->assertEquals(10,$tokenizer->numWords());
        $textDatas = $tokenizer->sequencesToTexts($liseqs);
        [$liseqs,$lilabels] = $this->sortResult($textDatas,$lilabels);
        $this->assertEquals('negative0 comment text',$liseqs[0]);
        // epoch 2
        $datas = [];
        foreach ($dataset as $key => $value) {
            $datas[] = $value;
        }
        $this->assertCount(3,$datas);

        $classnames = $dataset->classnames();
        sort($classnames);
        $this->assertEquals(['neg','pos'],$classnames);
        $classnames = $dataset->classnames();
        //public function loadData(string $filePath=null)

        [$inputs,$tests] = $dataset->loadData();
        $this->assertInstanceof(NDArray::class,$inputs);
        $this->assertInstanceof(NDArray::class,$tests);
        $this->assertEquals([5,4],$inputs->shape());
        $this->assertEquals([5],$tests->shape());
        $txts = $tokenizer->sequencesToTexts($inputs);
        $lbls = $this->int2label($tests,$classnames);
        $this->assertCount(5,$txts);
        [$results,$testResults] = $this->sortResult($txts,$lbls);
        $this->assertEquals([
            "negative0 comment text",
            "negative1 text",
            "positive0 message text",
            "positive1 some message text",
            "positive2 text",
        ],$results);
        $this->assertEquals(['neg','neg','pos','pos','pos'],$testResults);
    }

    public function testJustloaddata()
    {
        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);

        $dataset = $nn->data()->TextClassifiedDataset(
            __DIR__.'/../Dataset/text',
            [
                'pattern'=>'@.*\\.txt@',
                'batch_size'=>2,
                //'verbose'=>1,
            ]
        );

        [$inputs,$tests] = $dataset->loadData();
        $classnames = $dataset->classnames();
        $tokenizer = $dataset->getTokenizer();
        $this->assertInstanceof(NDArray::class,$inputs);
        $this->assertInstanceof(NDArray::class,$tests);
        $this->assertEquals([5,4],$inputs->shape());
        $this->assertEquals([5],$tests->shape());
        $txts = $tokenizer->sequencesToTexts($inputs);
        $lbls = $this->int2label($tests,$classnames);
        $this->assertCount(5,$txts);
        [$results,$testResults] = $this->sortResult($txts,$lbls);
        $this->assertEquals([
            "negative0 comment text",
            "negative1 text",
            "positive0 message text",
            "positive1 some message text",
            "positive2 text",
        ],$results);
        $this->assertEquals(['neg','neg','pos','pos','pos'],$testResults);
    }

    public function testLoadValidationData()
    {
        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);

        $dataset = $nn->data()->TextClassifiedDataset(
            __DIR__.'/../Dataset/text',
            [
                'pattern'=>'@.*\\.txt@',
                'batch_size'=>2,
                //'verbose'=>1,
            ]
        );

        [$inputs,$tests] = $dataset->loadData();
        $tokenizer = $dataset->getTokenizer();
        $classnames = $dataset->classnames();
        $val_dataset = new TextClassifiedDataset(
            $mo,
            __DIR__.'/../Dataset/text',
            [
                'pattern'=>'@.*\\.txt@',
                'batch_size'=>2,
                //'verbose'=>1,
                'tokenizer'=>$tokenizer,
                'classnames'=>$classnames,
            ]
        );

        [$val_inputs,$val_tests] = $val_dataset->loadData();

        $this->assertInstanceof(NDArray::class,$val_inputs);
        $this->assertInstanceof(NDArray::class,$val_tests);
        $this->assertEquals([5,4],$val_inputs->shape());
        $this->assertEquals([5],$val_tests->shape());
        $txts = $tokenizer->sequencesToTexts($val_inputs);
        $lbls = $this->int2label($val_tests,$classnames);
        $this->assertCount(5,$txts);
        [$results,$testResults] = $this->sortResult($txts,$lbls);
        $this->assertEquals([
            "negative0 comment text",
            "negative1 text",
            "positive0 message text",
            "positive1 some message text",
            "positive2 text",
        ],$results);
        $this->assertEquals(['neg','neg','pos','pos','pos'],$testResults);
    }
}
