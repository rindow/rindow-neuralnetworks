<?php
namespace Rindow\NeuralNetworks\Builder;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Data\Dataset\NDArrayDataset;
use Rindow\NeuralNetworks\Data\Dataset\SequentialDataset;
use Rindow\NeuralNetworks\Data\Dataset\CSVDataset;
use Rindow\NeuralNetworks\Data\Dataset\ClassifiedDirectoryDataset;
use Rindow\NeuralNetworks\Data\Dataset\Dataset;
use Rindow\NeuralNetworks\Data\Image\ImageFilter;
use Rindow\NeuralNetworks\Data\Image\ImageClassifiedDataset;
use Rindow\NeuralNetworks\Data\Sequence\TextClassifiedDataset;
use LogicException;

class Data
{
    protected object $matrixOperator;

    public function __construct(object $matrixOperator)
    {
        $this->matrixOperator = $matrixOperator;
    }

    public function __get( string $name ) : object
    {
        if(!method_exists($this,$name)) {
            throw new LogicException('Unknown dataset: '.$name);
        }
        return $this->$name();
    }

    public function __set( string $name, mixed $value ) : void
    {
        throw new LogicException('Invalid operation to set');
    }

    /**
     * @param array<NDArray>|NDArray $inputs
     */
    public function NDArrayDataset(array|NDArray $inputs, mixed ...$options) : object
    {
        return new NDArrayDataset($this->matrixOperator, $inputs, ...$options);
    }

    /**
     * @param iterable<NDArray|array{NDArray,NDArray}> $inputs
     */
    public function SequentialDataset(iterable $inputs, mixed ...$options) : object
    {
        return new SequentialDataset($this->matrixOperator, $inputs, ...$options);
    }

    public function CSVDataset(string $path, mixed ...$options) : object
    {
        return new CSVDataset($this->matrixOperator, $path, ...$options);
    }

    public function ImageFilter(mixed ...$options) : object
    {
        return new ImageFilter($this->matrixOperator, ...$options);
    }

    /**
     * @param Dataset<NDArray>|NDArray $dataset
     */
    public function ImageDataGenerator(Dataset|NDArray $dataset, mixed ...$options) : object
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
        if($dataset instanceof NDArray) {
            $dataset = new NDArrayDataset($this->matrixOperator, $dataset, ...$options);
        }
        $dataset->setFilter($filter);
        return $dataset;
    }

    public function ClassifiedDirectoryDataset(string $path, mixed ...$options) : object
    {
        return new ClassifiedDirectoryDataset($this->matrixOperator, $path, ...$options);
    }

    public function TextClassifiedDataset(string $path, mixed ...$options) : object
    {
        return new TextClassifiedDataset($this->matrixOperator, $path, ...$options);
    }

    public function ImageClassifiedDataset(string $path, mixed ...$options) : object
    {
        return new ImageClassifiedDataset($this->matrixOperator, $path, ...$options);
    }
}
