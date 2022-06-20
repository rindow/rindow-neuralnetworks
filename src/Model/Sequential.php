<?php
namespace Rindow\NeuralNetworks\Model;

use InvalidArgumentException;
use UnexpectedValueException;
use LogicException;
use Rindow\NeuralNetworks\Layer\LayerBase;
use Interop\Polite\Math\Matrix\NDArray;

class Sequential extends AbstractModel
{
    public function __construct($backend,$builder,$hda,array $layers=null)
    {
        parent::__construct($backend,$builder,$hda);
        if($layers!==null) {
            foreach ($layers as $layer) {
                $this->add($layer);
            }
        }
    }

    public function add($layer)
    {
        if(!($layer instanceof LayerBase)) {
            throw new InvalidArgumentException('invalid Layer');
        }
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

    protected function buildLayers(array $options=null) : void
    {
        // initialize weight paramators
        $inputShape = null;
        foreach ($this->layers as $layer) {
            $inputShape = $layer->build($inputShape);
            //$this->extractWeights($layer);
        }
        $this->params = [];
        $this->grads = [];
        foreach($this->layers as $weights) {
            $this->params = array_merge($this->params,$weights->getParams());
            $this->grads  = array_merge($this->grads, $weights->getGrads());
        }
    }

    public function forward(...$args)
    {
        $x = array_shift($args);
        $training = array_shift($args);
        $trues = array_shift($args);
        foreach($this->layers as $layer) {
            $x = $layer->forward($x, $training);
        }
        return $x;
    }

    protected function backward(array $dOutputs) : array
    {
        $dout = $dOutputs;
        $layers = array_reverse($this->layers);
        foreach ($layers as $layer) {
            $dout = $layer->backward($dout);
        }
        return $dout;
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
        $this->built = false;
        $this->params = [];
        $this->grads = [];
        $this->optimizer = null;
        $this->lossFunction = null;
        $this->metrics = null;
    }
}
