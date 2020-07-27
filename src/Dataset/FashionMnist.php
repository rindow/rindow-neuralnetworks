<?php
namespace Rindow\NeuralNetworks\Dataset;

class FashionMnist extends Mnist
{
    protected $urlBase = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/';
    
    protected function getDatasetDir()
    {
        return sys_get_temp_dir().'/rindow/nn/datasets/fashion-mnist';
    }
}
