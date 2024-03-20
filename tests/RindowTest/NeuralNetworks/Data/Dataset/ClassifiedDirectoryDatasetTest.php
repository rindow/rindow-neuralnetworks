<?php
namespace RindowTest\NeuralNetworks\Data\Dataset\ClassifiedDirectoryDatasetTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Data\Dataset\ClassifiedDirectoryDataset;

class ClassifiedDirectoryDatasetTest extends TestCase
{
    public function testNormal()
    {
        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);

        $dataset = new ClassifiedDirectoryDataset(
            $mo,
            __DIR__.'/text',
            pattern: '@.*\\.txt@',
            batch_size: 2,
        );
        $datas = [];
        $sets = [];
        foreach ($dataset as $key => $value) {
            $datas[] = $value;
            [$texts,$labels] = $value;
            foreach($texts as $key => $text) {
                $label = $labels[$key];
                $sets[trim($text).','.$label] = true;
            }
        }
        $this->assertCount(3,$datas);
        $this->assertCount(5,$sets);
        $this->assertEquals(3,count($dataset));
        $this->assertEquals(5,$dataset->datasetSize());
        $this->assertTrue($sets["negative0 comment text.,neg"]);
        $this->assertTrue($sets["negative1 text.,neg"]);
        $this->assertTrue($sets["positive0 message text.,pos"]);
        $this->assertTrue($sets["positive1 some message text.,pos"]);
        $this->assertTrue($sets["positive2 text.,pos"]);

        // epoch 2
        $datas = [];
        foreach ($dataset as $key => $value) {
            $datas[] = $value;
        }
        $this->assertCount(3,$datas);
    }

    public function testStreamMode()
    {
        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);

        $dataset = new ClassifiedDirectoryDataset(
            $mo,
            __DIR__.'/text',
            pattern: '@.*\\.txt@',
            batch_size: 0,
        );
        $sets = [];
        foreach ($dataset as $key => $value) {
            [$text,$label] = $value;
                $sets[trim($text).','.$label] = true;
        }
        $this->assertCount(5,$sets);
        $this->assertEquals(0,count($dataset));
        $this->assertEquals(5,$dataset->datasetSize());
        $this->assertTrue($sets["negative0 comment text.,neg"]);
        $this->assertTrue($sets["negative1 text.,neg"]);
        $this->assertTrue($sets["positive0 message text.,pos"]);
        $this->assertTrue($sets["positive1 some message text.,pos"]);
        $this->assertTrue($sets["positive2 text.,pos"]);

        // epoch 2
        $datas = [];
        foreach ($dataset as $key => $value) {
            $datas[] = $value;
        }
        $this->assertCount(5,$datas);
    }

    public function testUnclassifiedMode()
    {
        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);

        $dataset = new ClassifiedDirectoryDataset(
            $mo,
            __DIR__.'/text',
            pattern: '@.*\\.txt@',
            batch_size: 0,
            unclassified: true,
        );
        $sets = [];
        foreach ($dataset as $key => $value) {
                $sets[trim($value)] = true;
        }
        $this->assertCount(5,$sets);
        $this->assertEquals(0,count($dataset));
        $this->assertEquals(5,$dataset->datasetSize());
        $this->assertTrue($sets["negative0 comment text."]);
        $this->assertTrue($sets["negative1 text."]);
        $this->assertTrue($sets["positive0 message text."]);
        $this->assertTrue($sets["positive1 some message text."]);
        $this->assertTrue($sets["positive2 text."]);

        // epoch 2
        $datas = [];
        foreach ($dataset as $key => $value) {
            $datas[] = $value;
        }
        $this->assertCount(5,$datas);
    }
}
