<?php
namespace Rindow\NeuralNetworks\Builder;

use Rindow\NeuralNetworks\Dataset\Mnist;
use Rindow\NeuralNetworks\Dataset\FashionMnist;

class Datasets
{
    protected $matrixOperator;
    protected $mnist;
    protected $fashionMnist;

    public function __construct($matrixOperator)
    {
        $this->matrixOperator = $matrixOperator;
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
}
