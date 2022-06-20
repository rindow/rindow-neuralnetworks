<?php
namespace Rindow\NeuralNetworks\Builder;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Dataset\Mnist;
use Rindow\NeuralNetworks\Dataset\FashionMnist;
use Rindow\NeuralNetworks\Dataset\Cifar10;
use LogicException;

class Datasets
{
    protected $matrixOperator;
    protected $mnist;
    protected $fashionMnist;
    protected $cifar10;

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

    public function mnist()
    {
        if($this->mnist==null) {
            $this->mnist = new Mnist(
                $this->matrixOperator);

        }
        return $this->mnist;
    }

    public function fashionMnist()
    {
        if($this->fashionMnist==null) {
            $this->fashionMnist = new FashionMnist(
                $this->matrixOperator);

        }
        return $this->fashionMnist;
    }

    public function cifar10()
    {
        if($this->cifar10==null) {
            $this->cifar10 = new Cifar10(
                $this->matrixOperator);

        }
        return $this->cifar10;
    }
}
