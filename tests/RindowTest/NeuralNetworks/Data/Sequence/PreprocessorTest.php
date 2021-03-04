<?php
namespace RindowTest\NeuralNetworks\Data\Sequence\PreprocessorTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Data\Sequence\Preprocessor;

class Test extends TestCase
{
    public function testNormal()
    {
        $mo = new MatrixOperator();
        $prep = new Preprocessor($mo);
        $x = [
            [ 0, 1, 2],
            [10,11,12,13],
            [20,21,22,23,24],
        ];

        $y = $prep->padSequences($x);
        $this->assertEquals([
            [ 0, 0, 0, 1, 2],
            [ 0,10,11,12,13],
            [20,21,22,23,24],
        ],$y->toArray());
        $this->assertEquals(NDArray::int32,$y->dtype());

        $y = $prep->padSequences($x,['padding'=>'post']);
        $this->assertEquals([
            [ 0, 1, 2, 0, 0],
            [10,11,12,13, 0],
            [20,21,22,23,24],
        ],$y->toArray());

        $y = $prep->padSequences($x,['maxlen'=>4]);
        $this->assertEquals([
            [ 0, 0, 1, 2],
            [10,11,12,13],
            [21,22,23,24],
        ],$y->toArray());

        $y = $prep->padSequences($x,['maxlen'=>4,'truncating'=>'post']);
        $this->assertEquals([
            [ 0, 0, 1, 2],
            [10,11,12,13],
            [20,21,22,23],
        ],$y->toArray());

        $y = $prep->padSequences($x,['maxlen'=>4,'truncating'=>'post','padding'=>'post']);
        $this->assertEquals([
            [ 0, 1, 2, 0],
            [10,11,12,13],
            [20,21,22,23],
        ],$y->toArray());

        $y = $prep->padSequences($x,['maxlen'=>4,'padding'=>'post']);
        $this->assertEquals([
            [ 0, 1, 2, 0],
            [10,11,12,13],
            [21,22,23,24],
        ],$y->toArray());

        $y = $prep->padSequences($x,['value'=>99]);
        $this->assertEquals([
            [ 99,99, 0, 1, 2],
            [ 99,10,11,12,13],
            [20,21,22,23,24],
        ],$y->toArray());

        $y = $prep->padSequences($x,['value'=>0.5,'dtype'=>NDArray::float32]);
        $this->assertEquals([
            [ 0.5,0.5, 0, 1, 2],
            [ 0.5,10,11,12,13],
            [20,21,22,23,24],
        ],$y->toArray());
        $this->assertEquals(NDArray::float32,$y->dtype());
    }
}
