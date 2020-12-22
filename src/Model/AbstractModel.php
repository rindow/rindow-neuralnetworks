<?php
namespace Rindow\NeuralNetworks\Model;

use InvalidArgumentException;
use UnexpectedValueException;
use LogicException;
use Rindow\NeuralNetworks\Support\GenericUtils;
use Rindow\NeuralNetworks\Optimizer\Optimizer;
use Rindow\NeuralNetworks\Layer\LayerBase;
use Rindow\NeuralNetworks\Activation\Softmax;
use Rindow\NeuralNetworks\Activation\Sigmoid;
use Rindow\NeuralNetworks\Loss\Loss;
use Rindow\NeuralNetworks\Loss\SparseCategoricalCrossEntropy;
use Rindow\NeuralNetworks\Loss\CategoricalCrossEntropy;
use Rindow\NeuralNetworks\Loss\BinaryCrossEntropy;
use Rindow\NeuralNetworks\Callback\CallbackList;
use Interop\Polite\Math\Matrix\NDArray;

abstract class AbstractModel implements Model
{
    use GenericUtils;
    abstract protected function forwardStep(NDArray $inputs, NDArray $trues=null, bool $training=null) : NDArray;
    abstract protected function backwardStep(NDArray $dout) : NDArray;
    abstract protected function buildLayers(array $options=null) : void;

    protected $backend;
    protected $builder;
    protected $hda;
    protected $layers = [];
    protected $lastLayer;
    protected $optimizer;
    protected $metrics;
    protected $lossFunction;
    protected $params = [];
    protected $grads = [];
    protected $built = false;
    protected $shapeInspection=true;

    public function __construct($backend,$builder,$hda=null)
    {
        $this->backend = $backend;
        $this->builder = $builder;
        if($hda===null) {
            $this->hda = $builder->utils()->HDA();
        } else {
            $this->hda = $hda;
        }
    }

    protected function console($message)
    {
        if(defined('STDERR')) {
            fwrite(STDERR,$message);
        }
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

    public function layers() : array
    {
        return $this->layers;
    }

    protected function getLastLayer()
    {
        return $this->lastLayer;
    }

    protected function setLastLayer($lastLayer)
    {
        $this->lastLayer = $lastLayer;
    }

    public function compile(array $options=null) : void
    {
        extract($this->extractArgs([
            'optimizer'=>'SGD',
            'loss'=>'SparseCategoricalCrossEntropy',
            'metrics'=>['loss','accuracy'],
        ],$options));

        // resolve optimizer
        if(is_string($optimizer)) {
            $optimizer = strtolower($optimizer);
        }
        if($optimizer=='sgd') {
            $optimizer = $this->builder->optimizers()->Sgd();
        } elseif($optimizer=='adam') {
            $optimizer = $this->builder->optimizers()->Adam();
        }
        if(!($optimizer instanceof Optimizer)) {
            throw new InvalidArgumentException('invalid optimizer');
        }
        $this->optimizer = $optimizer;

        // resolve lastLoss Layer
        $lastLayer = $this->getLastLayer();
        if(!$lastLayer) {
            throw new InvalidArgumentException('no layer');
        }
        $activation = $lastLayer->getActivation();
        if(is_string($loss)) {
            $loss = strtolower($loss);
        }
        if($loss=='sparsecategoricalcrossentropy'||
            $loss=='sparse_categorical_crossentropy') {
            $loss = $this->builder->losses()->SparseCategoricalCrossEntropy();
        }
        if($loss instanceof SparseCategoricalCrossEntropy) {
            if($activation instanceof Softmax) {
                $loss->setFromLogits(true);
                $lastLayer->setActivation($loss);
            }
        } elseif($loss instanceof CategoricalCrossEntropy) {
            if($activation instanceof Softmax) {
                $loss->setFromLogits(true);
                $lastLayer->setActivation($loss);
            }
        } elseif($loss instanceof BinaryCrossEntropy) {
            if($activation instanceof Sigmoid) {
                $loss->setFromLogits(true);
                $lastLayer->setActivation($loss);
            }
        }
        if(!($loss instanceof Loss)) {
            throw new InvalidArgumentException('invalid loss function');
        }
        $this->lossFunction = $loss;

        // resolve metrics
        if(empty($metrics)) {
            $metrics = [];
        }
        $this->metrics = $metrics;
        $this->buildLayers($options);
        $layerNames = [];
        foreach ($this->layers as $layer) {
            $name = basename(str_replace('\\',DIRECTORY_SEPARATOR,get_class($layer)));
            if(isset($layerNames[$name])) {
                $i = 1;
                $base = $name;
                while(true) {
                    if(!isset($layerNames[$base.'_'.$i])) {
                        $name = $base.'_'.$i;
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

    protected function addWeights($weights)
    {
        if($weights instanceof LayerBase){
            $this->params = array_merge($this->params,$weights->getParams());
            $this->grads  = array_merge($this->grads, $weights->getGrads());
            return;
        }else{
            throw new InvalidArgumentException('invalid type to add weights');
        }
    }

    protected function registerLayer(LayerBase $layer,array $inputShape=null) : array
    {
        $this->layers[] = $layer;
        $outputShape = $layer->build($inputShape);
        $this->addWeights($layer);
        return $outputShape;
    }

    public function setShapeInspection(bool $enable)
    {
        if($this->shapeInspection==$enable)
            return;
        foreach ($this->layers as $layer) {
            $layer->setShapeInspection($enable);
        }
        $this->shapeInspection = $enable;
    }

    public function fit(NDArray $inputs, NDArray $tests, array $options=null) : array
    {
        if(!$this->built) {
            throw new LogicException('Not yet built');
        }
        $K = $this->backend;
        $mo = $K->localMatrixOperator();
        $localLA = $K->localLA();
        extract($this->extractArgs([
            'batch_size'=>32,
            'epochs'=>1,
            'verbose'=>1,
            'validation_data'=>null,
            'callbacks'=>null,
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
        $callbacks = new CallbackList($this,$callbacks);
        if($verbose>=1) {
            $this->console('Train on '.$inputCount.' samples');
            if($val_inputs) {
                $valInputCount = $val_inputs->shape()[0];
                $this->console(', validation on '.$valInputCount.' samples');
            }
            $this->console("\n");
        }
        $callbacks->onTrainBegin();
        for($epoch=0;$epoch<$epochs;$epoch++) {
            $callbacks->onEpochBegin($epoch);
            $startTime = time();
            if($verbose>=1) {
                $this->console('Epoch '.($epoch+1).'/'.$epochs." ");
            }
            if($batchIndexCount>1) {
                if($shuffle) {
                    $choice = $localLA->randomSequence($batchIndexCount);
                } else {
                    $choice = $mo->arange($batchIndexCount);
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
                $callbacks->onTrainBatchBegin($batchIndex);
                [$loss,$accuracy] = $this->trainingStep(
                    $choice[$batchIndex],$batch_size,$inputCount,$inputs,$tests,$shuffle,$this->metrics);
                $totalLoss += $loss;
                $totalAccuracy += $accuracy;
                $this->setShapeInspection(false);
                $callbacks->onTrainBatchEnd($batchIndex,['loss'=>$loss,'accuracy'=>$accuracy]);
            }

            if(in_array('loss',$this->metrics)) {
                $history['loss'][] = $totalLoss / $batchIndexCount;
            }
            if(in_array('accuracy',$this->metrics)) {
                $history['accuracy'][] = $totalAccuracy / $batchIndexCount;
            }
            $logs = ['loss'=>$totalLoss,'accuracy'=>$totalAccuracy];
            if($val_inputs) {
                [$loss, $accuracy] = $this->evaluate($val_inputs, $val_test,
                    ['batch_size'=>$batch_size,'verbose'=>0,'callbacks'=>$callbacks]);
                $history['val_loss'][] = $loss;
                $history['val_accuracy'][] = $accuracy;
                $logs = ['val_loss'=>$loss,'val_accuracy'=>$accuracy];
            }
            if($verbose>=1) {
                $sec = time() - $startTime;
                $this->console(' - '.$sec." sec.\n");
                foreach ($history as $key => $value) {
                    $this->console(' '.$key.':'.sprintf('%2.4f',array_pop($value)));
                }
                $this->console("\n");
            }
            $callbacks->onEpochEnd($epoch,$logs);
        }
        $this->setShapeInspection(true);
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
        $inputs = $K->array($inputs);
        $trues = $K->array($trues);

        if($shuffle) {
            $size = $inputs->shape()[0];
            if($size>1) {
                $choice = $K->randomSequence($size);
            } else {
                $choice = $K->zeros([1]);
            }
            $inputs = $K->select($inputs,$choice);
            $trues  = $K->select($trues,$choice);
        }

        $preds = $this->forwardStep($inputs, $trues, $training=true);
        $loss  = $this->loss($trues,$preds);
        if(is_nan($loss)) {
            throw new UnexpectedValueException("loss is unexpected value");
        }
        $this->backwardStep($this->lossFunction->differentiateLoss());

        if(in_array('accuracy',$metrics)) {
            //$preds = $this->forwardLastlayer($preds);
            $accuracy = $this->accuracy($trues,$preds);
        } else {
            $accuracy = 0;
        }

        $this->optimizer->update($this->params, $this->grads);
        return [$loss,$accuracy];
    }

    protected function loss(NDArray $trues,NDArray $preds) : float
    {
        return $this->lossFunction->loss($trues,$preds);
    }

    protected function accuracy(NDArray $trues,NDArray $preds) : float
    {
        return $this->lossFunction->accuracy($trues,$preds);
    }

    public function evaluate(NDArray $x, NDArray $t, array $options=null) : array
    {
        $K = $this->backend;
        extract($this->extractArgs([
            'batch_size'=>32,
            'verbose'=>0,
            'callbacks'=>null,
        ],$options));
        $totalLoss = 0.0;
        $totalAccuracy = 0.0;
        $inputCount = $x->shape()[0];
        $batchIndexCount = (int)ceil($inputCount / $batch_size);
        if(!($callbacks instanceof CallbackList)) {
            $callbacks = new CallbackList($this,$callbacks);
        }
        if($verbose>=1) {
            $startTime = time();
        }
        $callbacks->onTestBegin();
        for($batchIndex=0;$batchIndex<$batchIndexCount;$batchIndex++) {
            if($verbose>=1) {
                    $this->console('.');
            }
            $callbacks->onTestBatchBegin($batchIndex);
            $batchStart = $batchIndex*$batch_size;
            $batchEnd = ($batchIndex+1)*$batch_size-1;
            if($batchEnd>=$inputCount)
                $batchEnd = $inputCount-1;
            $inputs = $x[[$batchStart,$batchEnd]];
            $trues  = $t[[$batchStart,$batchEnd]];
            $inputs = $K->array($inputs);
            $trues = $K->array($trues);
            $preds = $this->forwardStep($inputs,$trues,$training=false);
            $loss  = $this->loss($trues,$preds);
            //$preds = $this->forwardLastlayer($preds);
            $accuracy = $this->accuracy($trues,$preds);
            $callbacks->onTestBatchEnd($batchIndex,['val_loss'=>$loss,'val_accuracy'=>$accuracy]);
            $totalLoss += $loss;
            $totalAccuracy += $accuracy;
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
        $callbacks->onTestEnd(['val_loss'=>$totalLoss,'val_accuracy'=>$totalAccuracy]);
        return [$totalLoss,$totalAccuracy];
    }

    public function predict(NDArray $inputs, array $options=null) : NDArray
    {
        extract($this->extractArgs([
            'callbacks'=>null,
        ],$options));

        if(!($callbacks instanceof CallbackList)) {
            $callbacks = new CallbackList($this,$callbacks);
        }
        $callbacks->onPredictBegin();
        $outputs = $this->forwardStep($inputs,$trues=null, $training=false);
        $callbacks->onPredictEnd();
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

    public function saveWeights(&$modelWeights,$portable=null) : void
    {
        $K = $this->backend;
        if(!isset($modelWeights['layers']))
            $modelWeights['layers'] = [];
        foreach($this->layers() as $layer) {
            if(!isset($modelWeights['layers'][$layer->getName()]))
                $modelWeights['layers'][$layer->getName()] = [];
            foreach($layer->getParams() as $idx => $param) {
                $param=$K->ndarray($param);
                if($portable)
                    $param = $this->converPortableSaveMode($param);
                $modelWeights['layers'][$layer->getName()][$idx] = serialize($param);
            }
        }
        $optimizer = $this->optimizer();
        if(!isset($modelWeights['optimizer']))
            $modelWeights['optimizer'] = [];
        foreach ($optimizer->getWeights() as $idx => $weights) {
            $weights=$K->ndarray($weights);
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
                $data = unserialize($weights[$idx]);
                $data = $K->array($data);
                $K->copy($data,$param);
            }
        }
        $optimizer = $this->optimizer();
        $optimizer->build($this->weights());
        foreach ($optimizer->getWeights() as $idx => $weights) {
            $data = unserialize($modelWeights['optimizer'][$idx]);
            $data = $K->array($data);
            $K->copy($data,$weights);
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
