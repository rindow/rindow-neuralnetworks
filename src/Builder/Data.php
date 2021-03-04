<?php
namespace Rindow\NeuralNetworks\Builder;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Data\Dataset\NDArrayDataset;
use Rindow\NeuralNetworks\Data\Dataset\CSVDataset;
use Rindow\NeuralNetworks\Data\Dataset\ClassifiedTextDataset;
use Rindow\NeuralNetworks\Data\Image\ImageFilter;
use Rindow\NeuralNetworks\Data\Image\ImageClassifiedDataset;
use Rindow\NeuralNetworks\Data\Sequence\TextClassifiedDataset;
use LogicException;

class Data
{
    protected $matrixOperator;

    public function __construct($matrixOperator)
    {
        $this->matrixOperator = $matrixOperator;
    }

    public function __get( string $name )
    {
        if(!method_exists($this,$name)) {
            throw new LogicException('Unknown dataset: '.$name);
        }
        return $this->$name();
    }

    public function __set( string $name, $value ) : void
    {
        throw new LogicException('Invalid operation to set');
    }

    public function NDArrayDataset(NDArray $inputs, array $options=null)
    {
        return new NDArrayDataset($this->matrixOperator, $inputs, $options);
    }

    public function CSVDataset(string $path, array $options=null)
    {
        return new CSVDataset($this->matrixOperator, $path, $options);
    }

    public function ImageFilter(array $options=null)
    {
        return new ImageFilter($this->matrixOperator, $options);
    }

    public function ImageDataGenerator(NDArray $inputs, array $options=null)
    {
        $leftargs = [];
        $filter = new ImageFilter($this->matrixOperator, $options, $leftargs);
        $leftargs['filter']=$filter;
        return new NDArrayDataset($this->matrixOperator, $inputs, $leftargs);
    }

    public function ClassifiedTextDataset(string $path, array $options=null)
    {
        return new ClassifiedTextDataset($this->matrixOperator, $path, $options);
    }

    public function TextClassifiedDataset(string $path, array $options=null)
    {
        return new TextClassifiedDataset($this->matrixOperator, $path, $options);
    }

    public function ImageClassifiedDataset(string $path, array $options=null)
    {
        return new ImageClassifiedDataset($this->matrixOperator, $path, $options);
    }
}
