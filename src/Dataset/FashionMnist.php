<?php
namespace Rindow\NeuralNetworks\Dataset;

class FashionMnist extends Mnist
{
    protected string $urlBase = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/';
    
    protected function getDatasetDir() : string
    {
        return $this->getRindowDatesetDir().'/fashion-mnist';
    }
}
