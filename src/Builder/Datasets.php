<?php
namespace Rindow\NeuralNetworks\Builder;

use Rindow\NeuralNetworks\Dataset\Mnist;

class Datasets
{
    protected $matrixOperator;
    protected $mnist;

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
}
