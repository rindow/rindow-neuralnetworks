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
use Rindow\NeuralNetworks\Data\Dataset\Dataset;
use Rindow\NeuralNetworks\Data\Dataset\NDArrayDataset;
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

    public function backend()
    {
        return $this->backend;
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

    public function fit($inputs, NDArray $tests=null, array $options=null) : array
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
            'filter'=>null,
        ],$options,$leftargs));
        if($inputs instanceof NDArray) {
            $options = [
                'batch_size'=>$batch_size,
                'shuffle'=>$shuffle,
                'filter'=>$filter,
            ];
            if($tests!==null) {
                $options['tests'] = $tests;
            }
            $dataset = new NDArrayDataset($K->localMatrixOperator(),$inputs,$options);
            $inputCount = count($inputs);
        } elseif($inputs instanceof Dataset) {
            if($tests!=null) {
                throw new InvalidArgumentException('The tests must be specified in the Dataset.');
            }
            $dataset = $inputs;
            $inputCount = $dataset->datasetSize();
        } else {
            throw new InvalidArgumentException('unsupported array type. inputs must be NDArray or Dataset.');
        }
        if($validation_data===null) {
            [$val_inputs, $val_test] = [null,null];
        } elseif(is_array($validation_data)) {
            [$val_inputs, $val_test] = $validation_data;
        } elseif($validation_data instanceof Dataset) {
            $val_inputs = $validation_data;
            $val_test = null;
        } else {
            throw new InvalidArgumentException('unsupported dataset type.'.
                ' validation_data must be set of NDArray or instance of Dataset.');
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
                if($val_inputs instanceof NDArray) {
                    $valInputCount = count($val_inputs);
                } elseif($val_inputs instanceof Dataset) {
                    $valInputCount = $val_inputs->datasetSize();
                } else {
                    throw new InvalidArgumentException('unsupported dataset type.'.
                        ' validation_data must be set of NDArray or instance of Dataset.');
                }
                $this->console(', validation on '.$valInputCount.' samples');
            }
            $this->console("\n");
        }
        $totalSteps = count($dataset);
        $callbacks->onTrainBegin();
        for($epoch=0;$epoch<$epochs;$epoch++) {
            $callbacks->onEpochBegin($epoch);
            $startTime = time();
            [$totalLoss,$totalAccuracy] =
                $this->trainProcess($dataset,$epoch,$epochs,$startTime,$totalSteps,
                                                    $verbose,$callbacks);
            if($totalSteps==0) {
                $totalSteps = count($dataset);
            }
            if($totalSteps==0) {
                $totalSteps=1;
            }
            if(in_array('loss',$this->metrics)) {
                $history['loss'][] = $totalLoss / $totalSteps;
            }
            if(in_array('accuracy',$this->metrics)) {
                $history['accuracy'][] = $totalAccuracy / $totalSteps;
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
                $this->console("- ${sec} sec.\n");
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

    protected function trainProcess(
        $dataset,$epoch,$epochs,$startTime,$totalSteps,$verbose,$callbacks)
    {
        $K = $this->backend;
        if($verbose>=1) {
            $this->console('Epoch '.($epoch+1).'/'.$epochs." ");
        }
        $totalLoss = 0;
        if($totalSteps==0) {
            $indicateCount = 1000;
        } else {
            $indicateCount = (int)($totalSteps/25);
        }
        $totalAccuracy = 0;
        $indicate = 0;
        foreach($dataset as $batchIndex => $data) {
            if($verbose>=1) {
                if($indicate==0) {
                    if($verbose==1) {
                        $this->progressBar($epoch,$epochs,$startTime,
                                $batchIndex,$totalSteps,25);
                    } else {
                        $this->console(".");
                    }
                }
                $indicate++;
                if($indicate>$indicateCount)
                    $indicate = 0;
            }
            $callbacks->onTrainBatchBegin($batchIndex);
            ////
            [$inputs,$trues] = $data;
            $inputs = $K->array($inputs);
            $trues = $K->array($trues);

            $preds = $this->forwardStep($inputs, $trues, $training=true);
            $loss  = $this->loss($trues,$preds);
            if(is_nan($loss)) {
                throw new UnexpectedValueException("loss is unexpected value");
            }
            $this->backwardStep($this->lossFunction->differentiateLoss());

            if(in_array('accuracy',$this->metrics)) {
                //$preds = $this->forwardLastlayer($preds);
                $accuracy = $this->accuracy($trues,$preds);
            } else {
                $accuracy = 0;
            }

            $this->optimizer->update($this->params, $this->grads);

            $totalLoss += $loss;
            $totalAccuracy += $accuracy;
            $this->setShapeInspection(false);
            $callbacks->onTrainBatchEnd($batchIndex,['loss'=>$loss,'accuracy'=>$accuracy]);
        }
        return [$totalLoss,$totalAccuracy];
    }

    protected function progressBar($epoch,$epochs,$startTime,$batchIndex,$batchIndexCount,$maxDot)
    {
        $epoch++;
        if($batchIndex==0) {
            $this->console("\rEpoch ${epoch}/${epochs} ");
            return;
        }
        $elapsed = time() - $startTime;
        if($batchIndexCount) {
            $completion = $batchIndex/$batchIndexCount;
            $estimated = $elapsed / $completion;
            $remaining = $estimated - $elapsed;
            $dot = (int)ceil($maxDot*$completion);
            $sec = $remaining % 60;
            $min = (int)floor($remaining/60) % 60;
            $hour = (int)floor($remaining/3600);
            $rem_string = ($hour?$hour.':':'').sprintf('%02d:%02d',$min,$sec);
        } else {
            $dot = 1;
            $rem_string = '????';
            $this->console($maxDot."\n");
        }
        $this->console("\rEpoch $epoch/$epochs [".str_repeat('.',$dot).str_repeat(' ',$maxDot-$dot).
            "] ${elapsed} sec. remain:${rem_string}  ");
    }

    protected function loss(NDArray $trues,NDArray $preds) : float
    {
        return $this->lossFunction->loss($trues,$preds);
    }

    protected function accuracy(NDArray $trues,NDArray $preds) : float
    {
        return $this->lossFunction->accuracy($trues,$preds);
    }

    public function evaluate($x, NDArray $t=null, array $options=null) : array
    {
        $K = $this->backend;
        extract($this->extractArgs([
            'batch_size'=>32,
            'verbose'=>0,
            'callbacks'=>null,
        ],$options));
        $totalLoss = 0.0;
        $totalAccuracy = 0.0;
        if(!($callbacks instanceof CallbackList)) {
            $callbacks = new CallbackList($this,$callbacks);
        }
        if($verbose>=1) {
            $startTime = time();
        }
        if($x instanceof NDArray) {
            $options = ['tests'=>$t,'batch_size'=>$batch_size];
            $dataset = new NDArrayDataset($K->localMatrixOperator(),$x,$options);
        } elseif($x instanceof Dataset) {
            if($t!=null) {
                throw new InvalidArgumentException('The tests must be specified in the Dataset.');
            }
            $dataset = $x;
        } else {
            throw new InvalidArgumentException('unsupported array type. inputs must be NDArray or Dataset.');
        }
        $callbacks->onTestBegin();
        foreach($dataset as $batchIndex => $data) {
            if($verbose>=1) {
                    $this->console('.');
            }
            $callbacks->onTestBatchBegin($batchIndex);
            [$inputs,$trues] = $data;
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
        $totalSteps = count($dataset);
        if($totalSteps==0) {
            $totalSteps=1;
        }
        $totalLoss = $totalLoss / $totalSteps;
        $totalAccuracy = $totalAccuracy / $totalSteps;
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
        $inputs = $this->backend->array($inputs);
        $callbacks->onPredictBegin();
        $outputs = $this->forwardStep($inputs,$trues=null, $training=false);
        $callbacks->onPredictEnd();
        $outputs = $this->backend->ndarray($outputs);
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
            echo substr(str_pad($layer->getName().'('.$type.')',29),0,29);
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
        foreach($this->weights() as $idx => $param) {
            $param=$K->ndarray($param);
            if($portable)
                $param = $this->converPortableSaveMode($param);
            $modelWeights['layers'][$idx] = serialize($param);
        }
        $optimizer = $this->optimizer();
        if(!isset($modelWeights['optimizer']))
            $modelWeights['optimizer'] = [];
        foreach ($optimizer->getWeights() as $idx => $weights) {
            $weights=$K->ndarray($weights);
            $modelWeights['optimizer'][$idx] = serialize($weights);
        }
    }

    public function loadWeights($modelWeights) : void
    {
        $K = $this->backend;
        foreach($this->weights() as $idx => $param) {
            $data = unserialize($modelWeights['layers'][$idx]);
            $data = $K->array($data);
            $K->copy($data,$param);
        }
        $optimizer = $this->optimizer();
        $optimizer->build($this->weights());
        foreach ($optimizer->getWeights() as $idx => $weights) {
            $data = unserialize($modelWeights['optimizer'][$idx]);
            $data = $K->array($data);
            $K->copy($data,$weights);
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
/*
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
*/
    public function save($filepath,$portable=null) : void
    {
        $f = $this->hda->open($filepath);
        $f['modelConfig'] = $this->toJson();
        $f['modelWeights'] = [];
        $this->saveWeights($f['modelWeights'],$portable);
    }

    public function saveWeightsToFile($filepath,$portable=null) : void
    {
        $f = $this->hda->open($filepath);
        $f['modelWeights'] = [];
        $this->saveWeights($f['modelWeights'],$portable);
    }

    public function loadWeightsFromFile($filepath)
    {
        $f = $this->hda->open($filepath,'r');
        $this->loadWeights($f['modelWeights']);
    }
}
