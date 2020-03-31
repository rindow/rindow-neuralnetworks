<?php
namespace Rindow\NeuralNetworks\Model;

use InvalidArgumentException;
use UnexpectedValueException;
use LogicException;
use Rindow\NeuralNetworks\Support\GenericUtils;
use Rindow\NeuralNetworks\Optimizer\Optimizer;
use Rindow\NeuralNetworks\Layer\Layer;
use Rindow\NeuralNetworks\Layer\Softmax;
use Rindow\NeuralNetworks\Layer\Sigmoid;
use Rindow\NeuralNetworks\Loss\Loss;
use Rindow\NeuralNetworks\Loss\SparseCategoricalCrossEntropy;
use Rindow\NeuralNetworks\Loss\CategoricalCrossEntropy;
use Rindow\NeuralNetworks\Loss\BinaryCrossEntropy;
use Interop\Polite\Math\Matrix\NDArray;

class Sequential
{
    use GenericUtils;
    protected $backend;
    protected $builder;
    protected $hda;
    protected $optimizer;
    protected $metrics;
    protected $layers = [];
    protected $lossFunction;
    protected $params = [];
    protected $grads = [];
    protected $built = false;

    public function __construct($backend,$builder,$hda,array $layers=null)
    {
        $this->backend = $backend;
        $this->builder = $builder;
        if($layers!==null) {
            foreach ($layers as $layer) {
                $this->add($layer);
            }
        }
        $this->hda = $hda;
    }

    protected function console($message)
    {
        if(defined('STDERR')) {
            fwrite(STDERR,$message);
        }
    }

    public function add($layer) : void
    {
        if(!($layer instanceof Layer)) {
            throw new InvalidArgumentException('invalid Layer');
        }
        $this->layers[] = $layer;
        //$activation = $layer->activation();
        //if($activation) {
        //    $this->layers[] = $activation;
        //}
    }

    public function compile(array $options=null) : void
    {
        extract($this->extractArgs([
            'optimizer'=>'SGD',
            'loss'=>'SparseCategoricalCrossEntropy',
            'metrics'=>['loss','accuracy'],
        ],$options));

        // resolve optimizer
        if($optimizer=='SGD') {
            $optimizer = $this->builder->optimizers()->Sgd();
        }
        if(!($optimizer instanceof Optimizer)) {
            throw new InvalidArgumentException('invalid optimizer');
        }
        $this->optimizer = $optimizer;

        // resolve lastLoss Layer
        $lastLayer = array_pop($this->layers);
        if(!$lastLayer) {
            throw new InvalidArgumentException('no layer');
        }
        if($loss=='SparseCategoricalCrossEntropy') {
            $loss = $this->builder->losses()->SparseCategoricalCrossEntropy();
        }
        if($loss instanceof SparseCategoricalCrossEntropy) {
            if($lastLayer instanceof Softmax) {
                $loss->setFromLogits(true);
                $lastLayer = $loss;
            }
        } elseif($loss instanceof CategoricalCrossEntropy) {
            if($lastLayer instanceof Softmax) {
                $loss->setFromLogits(true);
                $lastLayer = $loss;
            }
        } elseif($loss instanceof BinaryCrossEntropy) {
            if($lastLayer instanceof Sigmoid) {
                $loss->setFromLogits(true);
                $lastLayer = $loss;
            }
        }
        if(!($loss instanceof Loss)) {
            throw new InvalidArgumentException('invalid loss function');
        }
        array_push($this->layers,$lastLayer);
        $this->lossFunction = $loss;

        // resolve metrics
        if(empty($metrics)) {
            $metrics = [];
        }
        $this->metrics = $metrics;

        // initialize weight paramators
        $this->params = [];
        $this->grads = [];
        $inputShape = null;
        $layerNames = [];
        foreach ($this->layers as $layer) {
            $layer->build($inputShape);
            $this->params = array_merge($this->params,$layer->getParams());
            $this->grads  = array_merge($this->grads, $layer->getGrads());
            $inputShape = $layer->outputShape();

            $name = basename(str_replace('\\',DIRECTORY_SEPARATOR,get_class($layer)));
            if(isset($layerNames[$name])) {
                $i = 1;
                while(true) {
                    if(!isset($layers[$name.'_'.$i])) {
                        $name = $name.'_'.$i;
                        break;
                    }
                    $i++;
                }
            }
            $layerNames[$name] = true;
            $layer->setName($name);
        }

        $this->built = true;
    }

    public function layers() : array
    {
        return $this->layers;
    }

    public function lossFunction()
    {
        return $this->lossFunction;
    }

    public function optimizer()
    {
        return $this->optimizer;
    }

    public function weights() : array
    {
        return $this->params;
    }

    public function grads() : array
    {
        return $this->grads;
    }

    public function fit(NDArray $inputs, NDArray $tests, array $options=null) : array
    {
        if(!$this->built) {
            throw new LogicException('Not yet built');
        }
        $K = $this->backend;
        extract($this->extractArgs([
            'batch_size'=>32,
            'epochs'=>1,
            'verbose'=>1,
            'validation_data'=>null,
            'shuffle'=>true,
        ],$options));
        $inputCount = $inputs->shape()[0];
        $batchIndexCount = (int)ceil($inputCount / $batch_size);
        if($validation_data!==null) {
            [$val_inputs, $val_test] = $validation_data;
        } else {
            [$val_inputs, $val_test] = [null,null];
        }
        $history = ['loss'=>[], 'accuracy'=>[]];
        if($val_inputs) {
            $history['val_loss'] = [];
            $history['val_accuracy'] = [];
        }
        if($verbose>=1) {
            $this->console('Train on '.$inputCount.' samples');
            if($val_inputs) {
                $valInputCount = $val_inputs->shape()[0];
                $this->console(', validation on '.$valInputCount.' samples');
            }
            $this->console("\n");
        }
        for($epoch=0;$epoch<$epochs;$epoch++) {
            $startTime = time();
            if($verbose>=1) {
                $this->console('Epoch '.($epoch+1).'/'.$epochs." ");
            }
            if($batchIndexCount>1) {
                if($shuffle) {
                    $choice = $K->randomChoice($batchIndexCount,$batchIndexCount,false);
                } else {
                    $choice = $K->arange($batchIndexCount);
                }
            } else {
                $choice = [0];
            }
            $totalLoss = 0;
            $indicateCount = (int)($batchIndexCount/25);
            $totalAccuracy = 0;
            $indicate = 0;
            for($batchIndex=0;$batchIndex<$batchIndexCount;$batchIndex++) {
                if($verbose>=1) {
                    if($indicate==0)
                        $this->console('.');
                    $indicate++;
                    if($indicate>$indicateCount)
                        $indicate = 0;
                }
                [$loss,$accuracy] = $this->trainingStep(
                    $choice[$batchIndex],$batch_size,$inputCount,$inputs,$tests,$shuffle,$this->metrics);
                $totalLoss += $loss;
                $totalAccuracy += $accuracy;
            }

            if(in_array('loss',$this->metrics)) {
                $history['loss'][] = $totalLoss / $batchIndexCount;
            }
            if(in_array('accuracy',$this->metrics)) {
                $history['accuracy'][] = $totalAccuracy / $batchIndexCount;
            }
            if($val_inputs) {
                [$loss, $accuracy] = $this->evaluate($val_inputs, $val_test,
                    ['batch_size'=>$batch_size,'verbose'=>0]);
                $history['val_loss'][] = $loss;
                $history['val_accuracy'][] = $accuracy;
            }

            if($verbose>=1) {
                $sec = time() - $startTime;
                $this->console(' - '.$sec." sec.\n");
                foreach ($history as $key => $value) {
                    $this->console(' '.$key.':'.sprintf('%2.4f',array_pop($value)));
                }
                $this->console("\n");
            }
        }
        return $history;
    }

    protected function trainingStep($batchIndex,$batchSize,$inputCount,$x,$t,$shuffle,$metrics)
    {
        $K = $this->backend;
        $batchStart = $batchIndex*$batchSize;
        $batchEnd = ($batchIndex+1)*$batchSize-1;
        if($batchEnd>=$inputCount)
            $batchEnd = $inputCount-1;

        $inputs = $x[[$batchStart,$batchEnd]];
        $trues  = $t[[$batchStart,$batchEnd]];

        if($shuffle) {
            $size = $inputs->shape()[0];
            if($size>1) {
                $choice = $K->randomChoice($size,$size,false);
            } else {
                $choice = $K->zeros([1]);
            }
            $inputs = $K->select($inputs,$choice);
            $trues  = $K->select($trues,$choice);
        }

        $preds = $this->forwardStep($inputs, $training=true);
        $loss  = $this->lossFunction->loss($trues,$preds);
        if(is_nan($loss)) {
            throw new UnexpectedValueException("loss is unexpected value");
        }
        $this->backwardStep($this->lossFunction->differentiateLoss());

        if(in_array('accuracy',$metrics)) {
            //$preds = $this->forwardLastlayer($preds);
            $accuracy = $this->lossFunction->accuracy($trues,$preds);
        } else {
            $accuracy = 0;
        }

        $this->optimizer->update($this->params, $this->grads);
        return [$loss,$accuracy];
    }

    protected function forwardStep(NDArray $inputs, bool $training) : NDArray
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

    public function evaluate(NDArray $x, NDArray $t, array $options=null) : array
    {
        extract($this->extractArgs([
            'batch_size'=>32,
            'verbose'=>0,
        ],$options));
        $totalLoss = 0.0;
        $totalAccuracy = 0.0;
        $inputCount = $x->shape()[0];
        $batchIndexCount = (int)ceil($inputCount / $batch_size);
        if($verbose>=1) {
            $startTime = time();
        }
        for($batchIndex=0;$batchIndex<$batchIndexCount;$batchIndex++) {
            if($verbose>=1) {
                    $this->console('.');
            }
            $batchStart = $batchIndex*$batch_size;
            $batchEnd = ($batchIndex+1)*$batch_size-1;
            if($batchEnd>=$inputCount)
                $batchEnd = $inputCount-1;
            $inputs = $x[[$batchStart,$batchEnd]];
            $trues  = $t[[$batchStart,$batchEnd]];
            $preds = $this->forwardStep($inputs,$training=false);
            $totalLoss += $this->lossFunction->loss($trues,$preds);
            //$preds = $this->forwardLastlayer($preds);
            $totalAccuracy += $this->lossFunction->accuracy($trues,$preds);
        }
        $totalLoss = $totalLoss / $batchIndexCount;
        $totalAccuracy = $totalAccuracy / $batchIndexCount;
        if($verbose>=1) {
            $sec = time() - $startTime;
            $this->console(' - '.$sec." sec.\n");
            $this->console(' loss:'.sprintf('%2.4f',$totalLoss));
            $this->console(' accuracy:'.sprintf('%2.4f',$totalAccuracy));
            $this->console("\n");
        }
        return [$totalLoss,$totalAccuracy];
    }

    public function predict($inputs, array $options=null) : NDArray
    {
        //extract($this->extractArgs([
        //],$options));

        $outputs = $this->forwardStep($inputs, $training=false);
        return $outputs;
        //return $this->forwardLastlayer($outputs);
    }
/*
    protected function forwardLastlayer($x)
    {
        $layers = $this->layers;
        $lastLayer = array_pop($layers);
        if(method_exists($lastLayer,'incorporatedLoss') &&
            $lastLayer->incorporatedLoss()) {
                $lastLayer->setIncorporatedLoss(false);
                $x = $lastLayer->forward($x,false);
                $lastLayer->setIncorporatedLoss(true);
        }
        return $x;
    }
*/
    public function toJson() : string
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

    public function saveWeights(&$modelWeights,$portable=null) : void
    {
        $K = $this->backend;
        if(!isset($modelWeights['layers']))
            $modelWeights['layers'] = [];
        foreach($this->layers() as $layer) {
            if(!isset($modelWeights['layers'][$layer->getName()]))
                $modelWeights['layers'][$layer->getName()] = [];
            foreach($layer->getParams() as $idx => $param) {
                if($portable)
                    $param = $this->converPortableSaveMode($param);
                $modelWeights['layers'][$layer->getName()][$idx] = serialize($param);
            }
        }
        $optimizer = $this->optimizer();
        if(!isset($modelWeights['optimizer']))
            $modelWeights['optimizer'] = [];
        foreach ($optimizer->getWeights() as $idx => $weights) {
            $modelWeights['optimizer'][$idx] = serialize($weights);
        }
    }

    protected function converPortableSaveMode($ndarray) : NDArray
    {
        if($ndarray instanceof \Rindow\Math\Matrix\NDArrayPhp) {
            $ndarray = $ndarray->reshape($ndarray->shape());
            $ndarray->setPortableSerializeMode(true);
        }
        return $ndarray;
    }

    public function loadWeights($modelWeights) : void
    {
        $K = $this->backend;
        foreach($this->layers() as $layer) {
            $weights = $modelWeights['layers'][$layer->getName()];
            foreach($layer->getParams() as $idx => $param) {
                $K->copy(unserialize($weights[$idx]),$param);
            }
        }
        $optimizer = $this->optimizer();
        $optimizer->build($this->weights());
        foreach ($optimizer->getWeights() as $idx => $weights) {
            $K->copy(unserialize($modelWeights['optimizer'][$idx]),$weights);
        }
    }

    public function save($filepath,$portable=null) : void
    {
        $f = $this->hda->open($filepath);
        $f['modelConfig'] = $this->toJson();
        $f['modelWeights'] = [];
        $this->saveWeights($f['modelWeights'],$portable);
    }
}
