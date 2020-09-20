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

    public function add($layer) : void
    {
        if(!($layer instanceof LayerBase)) {
            throw new InvalidArgumentException('invalid Layer');
        }
        $this->layers[] = $layer;
        //$activation = $layer->activation();
        //if($activation) {
        //    $this->layers[] = $activation;
        //}
    }

    protected function getLastLayer()
    {
        $layers = $this->layers;
        $lastLayer = array_pop($layers);
        return $lastLayer;
    }

    protected function buildLayers(array $options=null) : void
    {
        // initialize weight paramators
        $this->params = [];
        $this->grads = [];
        $inputShape = null;
        foreach ($this->layers as $layer) {
            $inputShape = $layer->build($inputShape);
            $this->addWeights($layer);
        }
    }

    protected function forwardStep(NDArray $inputs, NDArray $trues=null, bool $training=null) : NDArray
    {
        $x = $inputs;
        foreach($this->layers as $layer) {
            $x = $layer->forward($x, $training);
        }
        return $x;
    }

    protected function backwardStep(NDArray $dout) : NDArray
    {
        $layers = array_reverse($this->layers);
        foreach ($layers as $layer) {
            $dout = $layer->backward($dout);
        }
        return $dout;
    }

    public function summary()
    {
        echo str_pad('Layer(type)',29).
            str_pad('Output Shape',27).
            str_pad('Param #',10)."\n";
        echo str_repeat('=',29+27+10)."\n";
        $totalParams = 0;
        foreach ($this->layers as $layer) {
            $type = basename(str_replace('\\',DIRECTORY_SEPARATOR,get_class($layer)));
            echo str_pad($layer->getName().'('.$type.')',29);
            echo str_pad('('.implode(',',$layer->outputShape()).')',27);
            $nump = 0;
            foreach($layer->getParams() as $p) {
                $nump += $p->size();
            }
            echo str_pad($nump,10);
            echo "\n";
            $totalParams += $nump;
        }
        echo str_repeat('=',29+27+10)."\n";
        echo 'Total params: '.$totalParams."\n";
    }
}
