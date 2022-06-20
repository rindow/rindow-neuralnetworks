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

    public function __construct(object $matrixOperator)
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

    public function NDArrayDataset($inputs, ...$options)
    {
        return new NDArrayDataset($this->matrixOperator, $inputs, ...$options);
    }

    public function CSVDataset(string $path, ...$options)
    {
        return new CSVDataset($this->matrixOperator, $path, ...$options);
    }

    public function ImageFilter(...$options)
    {
        return new ImageFilter($this->matrixOperator, ...$options);
    }

    public function ImageDataGenerator(NDArray $inputs, ...$options)
    {
        $data_format = $options['data_format'] ?? null;
        $height_shift = $options['height_shift'] ?? null;
        $width_shift = $options['width_shift'] ?? null;
        $vertical_flip = $options['vertical_flip'] ?? null;
        $horizontal_flip = $options['horizontal_flip'] ?? null;
        unset($options['data_format']);
        unset($options['height_shift']);
        unset($options['width_shift']);
        unset($options['vertical_flip']);
        unset($options['horizontal_flip']);

        $filter = new ImageFilter($this->matrixOperator, 
            data_format: $data_format,
            height_shift: $height_shift,
            width_shift: $width_shift,
            vertical_flip: $vertical_flip,
            horizontal_flip: $horizontal_flip,
        );
        $options['filter'] = $filter;
        return new NDArrayDataset($this->matrixOperator, $inputs, ...$options);
    }

    public function ClassifiedTextDataset(string $path, ...$options)
    {
        return new ClassifiedTextDataset($this->matrixOperator, $path, ...$options);
    }

    public function TextClassifiedDataset(string $path, ...$options)
    {
        return new TextClassifiedDataset($this->matrixOperator, $path, ...$options);
    }

    public function ImageClassifiedDataset(string $path, ...$options)
    {
        return new ImageClassifiedDataset($this->matrixOperator, $path, ...$options);
    }
}
