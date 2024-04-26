<?php
namespace Rindow\NeuralNetworks\Builder;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Dataset\Mnist;
use Rindow\NeuralNetworks\Dataset\FashionMnist;
use Rindow\NeuralNetworks\Dataset\Cifar10;
use LogicException;

class Datasets
{
    protected object $matrixOperator;
    protected ?object $mnist=null;
    protected ?object $fashionMnist=null;
    protected ?object $cifar10=null;

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

    public function mnist() : object
    {
        if($this->mnist==null) {
            $this->mnist = new Mnist(
                $this->matrixOperator);

        }
        return $this->mnist;
    }

    public function fashionMnist() : object
    {
        if($this->fashionMnist==null) {
            $this->fashionMnist = new FashionMnist(
                $this->matrixOperator);

        }
        return $this->fashionMnist;
    }

    public function cifar10() : object
    {
        if($this->cifar10==null) {
            $this->cifar10 = new Cifar10(
                $this->matrixOperator);

        }
        return $this->cifar10;
    }
}
