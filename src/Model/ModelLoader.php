<?php
namespace Rindow\NeuralNetworks\Model;

use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\NeuralNetworks\Support\HDA\HDAFactory;

class ModelLoader
{
    protected object $backend;
    protected Builder $builder;
    protected ?HDAFactory $hdaFactory;

    public function __construct(Builder $builder,HDAFactory $hdaFactory=null)
    {
        $this->builder = $builder;
        $this->backend = $builder->backend();
        $this->hdaFactory = $hdaFactory;
    }

    /**
     * @param array<mixed> $config
     * @return array<mixed>
     */
    protected function configToArgs(array $config) : array
    {
        $opts = $config['options'] ?? [];
        unset($config['options']);
        $args = array_merge(array_values($config),$opts);
        return $args;
    }

    /**
     * @param array<mixed> $config
     */
    public function modelFromConfig(array $config) : Model
    {
        $modelClass = $config['model']['class'];
        $model = new $modelClass($this->builder,$this->hdaFactory);
        $class = null;
        $layer = null;
        foreach($config['layer']['layers'] as $layerName => $layerConfig) {
            $class = $layerConfig['class'];
            if(isset($args['builder'])) {
                if($args['builder']){
                    $args['builder']=$this->builder;
                }else{
                    unset($args['builder']);
                }
            }
            //$args  = array_values($layerConfig['config']);
            $args  = $this->configToArgs($layerConfig['config']);
            $layer = new $class($this->backend,...$args);
            $layer->setName($layerName);
            $model->add($layer);
        }
        $lossFunctionName = $config['loss']['class'];
        if($class!=null && $layer!=null && $class===$lossFunctionName) {
            $loss = $layer;
        } else {
            //$args = array_values($config['loss']['config']);
            $args  = $this->configToArgs($config['loss']['config']);
            $loss = new $lossFunctionName($this->backend,...$args);
        }
        $optimizerName = $config['optimizer']['class'];
        //$args = array_values($config['optimizer']['config']);
        $args  = $this->configToArgs($config['optimizer']['config']);
        $optimizer = new $optimizerName($this->backend,...$args);
        $model->compile(
            loss: $loss,
            optimizer: $optimizer,
        );
        return $model;
    }

    public function loadModel(string $filepath) : Model
    {
        $f = $this->hdaFactory->open($filepath,'r');
        $config = json_decode($f['modelConfig'],true);
        $model = $this->modelFromConfig($config);
        $model->loadWeights($f['modelWeights']);
        return $model;
    }
}
