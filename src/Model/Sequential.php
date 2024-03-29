<?php
namespace Rindow\NeuralNetworks\Model;

use InvalidArgumentException;
use UnexpectedValueException;
use LogicException;
use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\NeuralNetworks\Gradient\Module;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Layer\Layer;
use Interop\Polite\Math\Matrix\NDArray;


class Sequential extends AbstractModel
{
    protected $layers = [];

    public function __construct(Builder $builder,$hda,array $layers=null)
    {
        parent::__construct($builder,$hda);
        if($layers!==null) {
            foreach ($layers as $layer) {
                $this->add($layer);
            }
        }
    }

    public function add(Module $layer)
    {
        $this->layers[] = $layer;
        //$this->extractWeights($layer);
        //$activation = $layer->activation();
        //if($activation) {
        //    $this->layers[] = $activation;
        //}
        return $layer;
    }

    //protected function getLastLayer()
    //{
    //    $layers = $this->layers;
    //    $lastLayer = array_pop($layers);
    //    return $lastLayer;
    //}

    public function layers() : array
    {
        $layers = [];
        foreach($this->layers as $module) {
            if($module instanceof Layer) {
                $layers[] = $module;
            } else {
                $layers = array_merge($layers,$module->layers());
            }
        }
        return $layers;
    }

    public function submodules() : array
    {
        return $this->layers;
    }
    
    public function variables() : array
    {
        $variables = [];
        foreach($this->layers as $module) {
            $variables = array_merge($variables,$module->variables());
        }

        return $variables;
    }
    
    protected function call($x, Variable|bool $training=null, ...$args)
    {
        $trainingOpt = ['training'=> $training];
        //$trues = array_shift($args);
        foreach($this->layers as $layer) {
            $options = [];
            if($layer->isAwareOf('training')) {
                $options = $trainingOpt;
            }
            $x = $layer->forward($x, ...$options);
        }
        return $x;
    }
/*
    protected function backward(array $dOutputs) : array
    {
        $dout = $dOutputs;
        $layers = array_reverse($this->layers);
        foreach ($layers as $layer) {
            $dout = $layer->backward($dout);
        }
        return $dout;
    }
*/
    protected function generateLayersConfig() : array
    {
        $layerNames = [];
        $layers = [];

        foreach ($this->layers as $layer) {
            $name = $layer->getName();
            $layerNames[] = $name;
            $layers[$name] = [
                'class'  => get_class($layer),
                'config' => $layer->getConfig(),
            ];
        }
        return [$layerNames,$layers];
    }

    public function summary()
    {
        if(!$this->built) {
            $first = $this->layers[0];
            $inputShape = $first->inputShape();
            array_unshift($inputShape,1);
            $this->build($inputShape);
        }
        parent::summary();
    }

    public function toJson() : string
    {
        [$layerNames,$layers] = $this->generateLayersConfig();

        $modelConfig = [
            'model' => [
                'class' => get_class($this),
            ],
            'layer' => [
                'layerNames' => $layerNames,
                'layers' => $layers,
            ],
            'loss' => [
                'class' => get_class($this->lossFunction),
                'config' => $this->lossFunction->getConfig(),
            ],
            'optimizer' => [
                'class' => get_class($this->optimizer),
                'config' => $this->optimizer->getConfig(),
            ],
        ];
        $configJson = json_encode($modelConfig,JSON_UNESCAPED_SLASHES);
        return $configJson;
    }

    public function save($filepath,$portable=null) : void
    {
        $f = $this->hda->open($filepath);
        $f['modelConfig'] = $this->toJson();
        $f['modelWeights'] = [];
        $this->saveWeights($f['modelWeights'],$portable);
    }

    public function __clone()
    {
        $newLayers = [];
        foreach ($this->layers as $layer) {
            $newLayers[] = clone $layer;
        }
        $this->layers = $newLayers;
        //$this->built = false;
        //$this->params = [];
        //$this->grads = [];
        //$this->optimizer = null;
        //$this->lossFunction = null;
        //$this->metrics = null;
    }
}
