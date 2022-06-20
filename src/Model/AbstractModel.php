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
use Rindow\NeuralNetworks\Gradient\Core\Variable;
use Rindow\NeuralNetworks\Gradient\Core\Undetermined;
use Rindow\NeuralNetworks\Gradient\Core\UndeterminedNDArray;
use Interop\Polite\Math\Matrix\NDArray;

abstract class AbstractModel implements Model
{
    use GenericUtils;

    protected $backend;
    protected $builder;
    protected $hda;
    protected $name;
    protected $layers = [];
    protected $lastLayer;
    protected $optimizer;
    protected $metrics;
    protected $lossFunction;
    protected $params = [];
    protected $grads = [];
    protected $built = false;
    protected $shapeInspection=true;
    protected $inputsVariables;
    protected $outputsVariables;

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

    public function setName($name)
    {
        $this->name = $name;
    }

    public function name()
    {
        return $this->name;
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

    public function params() : array
    {
        return $this->params;
    }

    public function grads() : array
    {
        return $this->grads;
    }

    //protected function getLastLayer()
    //{
    //    return $this->lastLayer;
    //}

    //protected function setLastLayer($lastLayer)
    //{
    //    $this->lastLayer = $lastLayer;
    //}
    protected function basename($object) : string
    {
        $classname = get_class($object);
        return substr($classname,strrpos($classname,'\\')+1);
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
            if(is_string($optimizer)) {
                $msg = $optimizer;
            } elseif(is_object($optimizer)) {
                $msg = get_class($optimizer);
            } else {
                $msg = gettype($optimizer);
            }
            throw new InvalidArgumentException('invalid optimizer: '.$msg);
        }
        $this->optimizer = $optimizer;

        if(is_string($loss)) {
            $loss = strtolower($loss);
        }
        if($loss=='sparsecategoricalcrossentropy'||
            $loss=='sparse_categorical_crossentropy') {
            $loss = $this->builder->losses()->SparseCategoricalCrossEntropy();
        }

        if(!($loss instanceof Loss)) {
            if(is_string($loss)) {
                $msg = $loss;
            } elseif(is_object($loss)) {
                $msg = get_class($loss);
            } else {
                $msg = gettype($loss);
            }
            throw new InvalidArgumentException('invalid loss function: '.$msg);
        }
        $this->lossFunction = $loss;

        // resolve metrics
        if(empty($metrics)) {
            $metrics = [];
        }
        $this->metrics = $metrics;

        // build pipeline of layers
        $this->buildLayers($options);

        $layerNames = [];
        foreach ($this->layers() as $layer) {
            $name = $this->basename($layer);
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

            [$loss, $accuracy] = $this->trainStep($inputs, $trues);
            $totalLoss += $loss;
            $totalAccuracy += $accuracy;
            if($this->shapeInspection) {
                $this->setShapeInspection(false);
            }
            $callbacks->onTrainBatchEnd($batchIndex,['loss'=>$loss,'accuracy'=>$accuracy]);
        }
        if($verbose==1) {
            $this->progressBar($epoch,$epochs,$startTime,
                    $totalSteps,$totalSteps,25);
        } elseif($verbose>1) {
            $this->console(".");
        }
        return [$totalLoss,$totalAccuracy];
    }

    protected function trainStep($inputs, $trues)
    {
        $preds = $this->forward($inputs, $training=true, $trues);
        $trues = $this->trueValuesFilter($trues);
        $loss  = $this->loss($trues,$preds);
        if(is_nan($loss)) {
            throw new UnexpectedValueException("loss is unexpected value");
        }
        //$this->backwardStep($this->lossFunction->differentiateLoss());
        $this->backward($this->lossFunction->backward([1]));

        if(in_array('accuracy',$this->metrics)) {
            //$preds = $this->forwardLastlayer($preds);
            $accuracy = $this->accuracy($trues,$preds);
        } else {
            $accuracy = 0;
        }

        $this->optimizer->update($this->params, $this->grads);
        return [$loss,$accuracy];
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
            "] ${elapsed} sec. remaining:${rem_string}  ");
    }

    protected function trueValuesFilter(NDArray $trues) : NDArray
    {
        return $trues;
    }

    protected function loss(NDArray $trues,NDArray $preds) : float
    {
        return $this->lossFunction->forward($trues,$preds);
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

            [$loss,$accuracy] = $this->evaluateStep($inputs,$trues);

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

    protected function evaluateStep($inputs,$trues)
    {
        $preds = $this->forward($inputs,$training=false,$trues);
        $trues = $this->trueValuesFilter($trues);
        $loss  = $this->loss($trues,$preds);
        //$preds = $this->forwardLastlayer($preds);
        $accuracy = $this->accuracy($trues,$preds);
        return [$loss,$accuracy];
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
        $outputs = $this->predictStep($inputs,$options);
        $callbacks->onPredictEnd();
        $outputs = $this->backend->ndarray($outputs);
        return $outputs;
        //return $this->forwardLastlayer($outputs);
    }

    protected function predictStep($inputs,$options)
    {
        return $this->forward($inputs, $training=false, $trues=null);
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
    public function inputs()
    {
        return $this->inputsVariables;
    }

    public function outputs()
    {
        return $this->outputsVariables;
    }

    /*
    *  static step interfaces
    */
    //protected function extractWeights($weights)
    //{
    //    if($weights instanceof LayerBase){
    //        $this->params = array_merge($this->params,$weights->getParams());
    //        $this->grads  = array_merge($this->grads, $weights->getGrads());
    //        return;
    //    }else{
    //        throw new InvalidArgumentException('invalid type to add weights');
    //    }
    //}

    //protected function registerLayer(LayerBase $layer,array $inputShape=null) : array
    //{
    //    $this->layers[] = $layer;
    //    $outputShape = $layer->build($inputShape);
    //    $this->extractWeights($layer);
    //    return $outputShape;
    //}

    protected function buildLayers(array $options=null) : void
    {
        $nn = $this->builder;
        $inputs = new Undetermined();
        $trues = new Undetermined();
        $model = $this;
        $outputs = $nn->with($ctx=new BuildContext(),
            function() use ($model,$inputs,$trues) {
                $outputs = $model->forward($inputs,true,$trues);
                return $outputs;
            }
        );
        if(!($outputs instanceof Variable)) {
            if(is_object($outputs)) {
                $type = get_class($outputs);
            } else {
                $type = gettype($outputs);
            }
            throw new LogicException('root model must output single Variable: '.$type);
        }
        $this->outputsVariables = [$outputs->reference()];

        $funcs = [$outputs->creator()];
        $pipeline = [];
        $used = [];

        while(count($funcs)>0) {
            $func = array_pop($funcs);
            $pipeline[] = $func;
            foreach($func->inputs() as $input) {
                $creator = $input->creator();
                if($creator!=null) {
                    $oid = spl_object_hash($creator);
                    if(!array_key_exists($oid,$used)) {
                        $used[$oid] = true;
                        $funcs[] = $creator;
                        usort($funcs,function($a,$b){return $a->generation()-$b->generation();});
                    }
                }
            }
        }
        $this->pipeline = [];
        foreach($pipeline as $func) {
            $oid = spl_object_hash($func);
            $this->pipeline[$oid] = $func;
        }
        $this->layers = [];
        foreach ($ctx->getList() as $layer) {
            $oid = spl_object_hash($layer);
            if(array_key_exists($oid,$this->pipeline)) {
                $this->layers[] = $layer;
            }
        }
        $this->params = [];
        $this->grads = [];
        foreach($this->layers as $weights) {
            $this->params = array_merge($this->params,$weights->getParams());
            $this->grads  = array_merge($this->grads, $weights->getGrads());
        }
    }

    public function forward(...$inputs)
    {
        return $this->call(...$inputs);
    }

    protected function backward(array $dOutputs) : array
    {
        $K = $this->backend;
        if(count($dOutputs)!=1) {
            throw new InvalidArgumentException('The dOutputs must be a list containing one NDArray.');
        }
        [$dOutputs] = $dOutputs;
        if(!($dOutputs instanceof NDArray)) {
            throw new InvalidArgumentException('The dOutputs must be a list containing one NDArray.');
        }
        $batchSize = $dOutputs->shape()[0];
        $oid = $this->outputs()[0]->oid();
        $grads[$oid] = $dOutputs;
        unset($dOutputs);
        $funcs = $this->pipeline;
        foreach ($funcs as $func) {
            $dOutputs = [];
            foreach($func->outputs() as $o) {
                $oid = $o->oid();
                if(array_key_exists($oid,$grads)) {
                    $dOutputs[] = $grads[$oid];
                    // *** CAUTION ***
                    // Outputs are not used after being used backwards.
                    // grads should also be released. See dynamic mode.
                    unset($grads[$oid]);
                } else {
                    $shape = $o->valueShape();
                    $dtype = $o->dtype();
                    array_unshift($shape,$batchSize);
                    $dOutputs[] = $K->zeros($shape,$dtype);
                }
            }
            $tmpdInputs = $func->backward($dOutputs);
            unset($dOutputs);

            $dDatas = array_map(null,$func->inputs(),$tmpdInputs);
            unset($tmpdInputs);

            foreach ($dDatas as $dData) {
                [$dInputs,$dx] = $dData;
                $oid = spl_object_hash($dInputs);
                if(array_key_exists($oid,$grads)) {
                    $K->update_add($grads[$oid],$dx);
                } else {
                    $grads[$oid] = $dx;
                }
            }
        }
        return []; // No reverse
    }

    public function layers() : array
    {
        $layers = [];
        foreach ($this->layers as $layer) {
            if($layer instanceof LayerBase) {
                $layers[] = $layer;
            } else {
                $layers = array_merge($layers,$layer->layers());
            }
        }
        return $layers;
    }

    public function parameterVariables() : array
    {
        return [];
    }

    public function summary()
    {
        echo str_pad('Layer(type)',29).
            str_pad('Output Shape',27).
            str_pad('Param #',10)."\n";
        echo str_repeat('=',29+27+10)."\n";
        $totalParams = 0;
        foreach ($this->layers() as $layer) {
            $type = $this->basename($layer);
            echo substr(str_pad($layer->getName().'('.$type.')',29),0,29);
            $outputShape = $layer->outputShape();
            if(is_array($outputShape[0])) {
                $outputShape = $outputShape[0];
            }
            echo str_pad('('.implode(',',$outputShape).')',27);
            $nump = 0;
            foreach($layer->getParams() as $p) {
                $nump += $p->size();
            }
            echo str_pad($nump,10);
            echo "\n";
            $totalParams += $nump;
        }
        $params = $this->parameterVariables();
        if(count($params)) {
            echo str_repeat('=',29+27+10)."\n";
            echo str_pad('Weights',29).
                str_pad('Shape',27).
                str_pad('Param #',10)."\n";
            echo str_repeat('=',29+27+10)."\n";
            foreach($params as $param) {
                $name = $param->name();
                if(!$name) {
                    $name = 'No name';
                }
                echo str_pad($name,29);
                echo str_pad('('.implode(',',$param->shape()).')',27);
                $nump = $param->size();
                echo str_pad($nump,10);
                echo "\n";
                $totalParams += $nump;
            }
        }
        echo str_repeat('=',29+27+10)."\n";
        echo 'Total params: '.$totalParams."\n";
    }

    protected function generateLayersConfig() : array
    {
        $layerNames = [];
        $layers = [];

        foreach ($this->layers() as $layer) {
            $name = $layer->getName();
            $layerNames[] = $name;
            $layers[$name] = [
                'class'  => get_class($layer),
                'config' => $layer->getConfig(),
            ];
        }
        return [$layerNames,$layers];
    }

    public function saveWeights(&$modelWeights,$portable=null) : void
    {
        $K = $this->backend;
        if(!isset($modelWeights['layers']))
            $modelWeights['layers'] = [];
        foreach($this->params() as $idx => $param) {
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
*/

    public function loadWeights($modelWeights) : void
    {
        $K = $this->backend;
        foreach($this->params() as $idx => $param) {
            $data = unserialize($modelWeights['layers'][$idx]);
            $data = $K->array($data);
            $K->copy($data,$param);
        }
        $optimizer = $this->optimizer();
        $optimizer->build($this->params());
        foreach ($optimizer->getWeights() as $idx => $weights) {
            $data = unserialize($modelWeights['optimizer'][$idx]);
            $data = $K->array($data);
            $K->copy($data,$weights);
        }
    }
/*
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
        $optimizer->build($this->params());
        foreach ($optimizer->getWeights() as $idx => $weights) {
            $data = unserialize($modelWeights['optimizer'][$idx]);
            $data = $K->array($data);
            $K->copy($data,$weights);
        }
    }
*/
    protected function converPortableSaveMode($ndarray) : NDArray
    {
        if($ndarray instanceof \Rindow\Math\Matrix\NDArrayPhp) {
            $ndarray = $ndarray->reshape($ndarray->shape());
            $ndarray->setPortableSerializeMode(true);
        }
        return $ndarray;
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
