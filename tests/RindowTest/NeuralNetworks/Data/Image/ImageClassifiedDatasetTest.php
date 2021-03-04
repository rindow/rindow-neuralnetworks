<?php
namespace RindowTest\NeuralNetworks\Data\Image\ImageClassifiedDatasetTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Data\Image\ImageClassifiedDataset;
use Rindow\Math\Plot\Plot;

class Test extends TestCase
{
    public function getPlotConfig()
    {
        return [
            'renderer.skipCleaning' => true,
            'renderer.skipRunViewer' => getenv('TRAVIS_PHP_VERSION') ? true : false,
        ];
    }

    public function testNormal()
    {
        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $dataset = $nn->data()->ImageClassifiedDataset(
            __DIR__.'/image',
            [
                'pattern'=>'@.*\\.png@',
                'batch_size'=>2,
            ]
        );
        $steps = 0;
        $totalsize = 0;
        $classnames = ['a','b'];
        [$fig,$axes] = $plt->subplots(2,2);
        foreach ($dataset as $step => $value) {
            [$images,$labels] = $value;
            $steps++;
            foreach($images as $key => $image) {
                $label = $labels[$key];
                $axes[$totalsize]->imshow($image,null,null,null,$origin='upper');
                $axes[$totalsize]->setTitle($classnames[$label]);
                $totalsize++;
            }
        }
        $plt->show();
        $this->assertCount(2,$dataset->classnames());
        $this->assertEquals(2,$steps);
        $this->assertEquals(4,$totalsize);
    }

    public function testStreamMode()
    {
        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $dataset = $nn->data()->ImageClassifiedDataset(
            __DIR__.'/image',
            [
                'pattern'=>'@.*\\.png@',
                'batch_size'=>0,
            ]
        );
        $totalsize = 0;
        [$fig,$axes] = $plt->subplots(2,2);
        foreach ($dataset as $key => $value) {
            [$image,$label] = $value;
            $axes[$totalsize]->imshow($image,null,null,null,$origin='upper');
            $axes[$totalsize]->setTitle($label);
            $totalsize++;
        }
        $plt->show();
        $this->assertEquals(4,$totalsize);

        // epoch 2
        $datas = [];
        foreach ($dataset as $key => $value) {
            $datas[] = $value;
        }
        $this->assertCount(4,$datas);
    }

    public function testUnclassifiedMode()
    {
        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $dataset = $nn->data()->ImageClassifiedDataset(
            __DIR__.'/image',
            [
                'pattern'=>'@.*\\.png@',
                'batch_size'=>0,
                'unclassified'=>true,
            ]
        );
        $totalsize = 0;
        [$fig,$axes] = $plt->subplots(2,2);
        foreach ($dataset as $key => $image) {
                $axes[$totalsize]->imshow($image,null,null,null,$origin='upper');
                $totalsize++;
        }
        $plt->show();
        $this->assertEquals(4,$totalsize);

        // epoch 2
        $datas = [];
        foreach ($dataset as $key => $value) {
            $datas[] = $value;
        }
        $this->assertCount(4,$datas);
    }

    public function testLoadData()
    {
        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $dataset = $nn->data()->ImageClassifiedDataset(
            __DIR__.'/image',
            [
                'pattern'=>'@.*\\.png@',
                'batch_size'=>2,
            ]
        );
        [$inputs,$tests] = $dataset->loadData();
        $this->assertInstanceof(NDArray::class,$inputs);
        $this->assertInstanceof(NDArray::class,$tests);
        $classnames = $dataset->classnames();
        $totalsize = 0;
        [$fig,$axes] = $plt->subplots(2,2);
        foreach($inputs as $key => $image) {
            $label = $tests[$key];
            $axes[$totalsize]->imshow($image,null,null,null,$origin='upper');
            $axes[$totalsize]->setTitle($classnames[$label]);
            $totalsize++;
        }
        $plt->show();
        $this->assertEquals(4,$totalsize);
    }
}
