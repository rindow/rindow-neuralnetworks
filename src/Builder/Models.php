<?php
namespace Rindow\NeuralNetworks\Builder;

use Rindow\NeuralNetworks\Model\Sequential;
use Rindow\NeuralNetworks\Model\ModelLoader;
use Rindow\NeuralNetworks\Model\Model;
use Rindow\NeuralNetworks\Layer\Layer;

class Models
{
    protected Builder $builder;

    public function __construct(Builder $builder)
    {
        $this->builder = $builder;
    }

    /**
     * @param array<Layer> $layers
     */
    public function Sequential(?array $layers=null) : object
    {
        return new Sequential($this->builder,
                        $this->builder->utils()->HDA(),$layers);
    }

    public function loadModel(mixed $filepath) : object
    {
        $loader = new ModelLoader($this->builder,
                                        $this->builder->utils()->HDA());
        return $loader->loadModel($filepath);
    }

    public function modelFromConfig(mixed $config) : object
    {
        $loader = new ModelLoader($this->builder,
                                        $this->builder->utils()->HDA());
        return $loader->modelFromConfig($config);
    }
}
