<?php
namespace Rindow\NeuralNetworks\Model;

use InvalidArgumentException;
use UnexpectedValueException;
use LogicException;
use ReflectionClass;
use ArrayAccess;
use Rindow\NeuralNetworks\Optimizer\Optimizer;
use Rindow\NeuralNetworks\Layer\Layer;
use Rindow\NeuralNetworks\Activation\Softmax;
use Rindow\NeuralNetworks\Activation\Sigmoid;
use Rindow\NeuralNetworks\Loss\Loss;
use Rindow\NeuralNetworks\Loss\SparseCategoricalCrossEntropy;
use Rindow\NeuralNetworks\Loss\CategoricalCrossEntropy;
use Rindow\NeuralNetworks\Loss\BinaryCrossEntropy;
use Rindow\NeuralNetworks\Callback\CallbackList;
use Rindow\NeuralNetworks\Data\Dataset\Dataset;
use Rindow\NeuralNetworks\Data\Dataset\NDArrayDataset;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Gradient\Module;
use Rindow\NeuralNetworks\Gradient\Core\GradientTape;
use Interop\Polite\Math\Matrix\NDArray;

abstract class AbstractModel implements Model
{
    protected $backend;
    protected $builder;
    protected $hda;
    protected $name;
    protected $optimizer;
    protected $metrics;
    protected $lossFunction;
    protected $built = false;
    protected $shapeInspection=true;
    protected $backupShapeInspection;
    protected $inputsVariables;
    protected $outputsVariables;
    protected $weightVariables = [];
    protected $trainableVariables;
    protected $generation;
    protected $graph = [];
    protected $weights;

    public function __construct(object $backend,object $builder,$hda=null)
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

    protected function display($message)
    {
        echo $message;
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

    protected function basename($object) : string
    {
        $classname = get_class($object);
        return substr($classname,strrpos($classname,'\\')+1);
    }

    public function compile(
        string|object $optimizer=null,
        string|object $loss=null,
        array $metrics=null,
        int $numInputs=null,
    ) : void
    {
        $optimizer = $optimizer ?? 'SGD';
        $loss = $loss ?? 'SparseCategoricalCrossEntropy';
        $metrics = $metrics ?? ['loss','accuracy'];
        $numInputs = $numInputs ?? 1;

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

    }

    public function fit(
        $inputs,
        NDArray $tests=null,
        int $batch_size=null,
        int $epochs=null,
        int $verbose=null,
        array|Dataset $validation_data=null,
        array $callbacks=null,
        bool $shuffle=null,
        object $filter=null,
    ) : array
    {
        if($this->optimizer==null || $this->lossFunction==null) {
            throw new LogicException('Not yet compile');
        }
        $K = $this->backend;
        $mo = $K->localMatrixOperator();
        $localLA = $K->localLA();

        // defaults
        $batch_size = $batch_size ?? 32;
        $epochs = $epochs ?? 1;
        $verbose = $verbose ?? 1;
        $validation_data = $validation_data ?? null;
        $callbacks = $callbacks ?? null;
        $shuffle = $shuffle ?? true;
        $filter = $filter ?? null;

        if($inputs instanceof NDArray||is_array($inputs)) {
            $dataset = new NDArrayDataset(
                $K->localMatrixOperator(),
                $inputs,
                batch_size: $batch_size,
                shuffle: $shuffle,
                filter: $filter,
                tests: $tests,
            );
            $inputCount = $dataset->datasetSize();
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
        $this->backupShapeInspection = $this->shapeInspection;
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
                    batch_size:$batch_size, verbose:0, callbacks:$callbacks);
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
        if($this->backupShapeInspection) {
            $this->setShapeInspection(true);
        }
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
            if(!is_array($inputs)) {
                $inputs = $K->array($inputs);
            } else {
                $newInputs = [];
                foreach ($inputs as $value) {
                    $newInputs[] = $K->array($value);
                }
                $inputs = $newInputs;
                unset($newInputs);
            }
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

    public function evaluate(
        $inputs,
        NDArray $trues=null,
        int $batch_size=null,
        int $verbose=null,
        object|array $callbacks=null,
    ) : array
    {
        // defaults
        $batch_size = $batch_size ?? 32;
        $verbose = $verbose ?? 0;
        $callbacks = $callbacks ?? null;

        $x = $inputs; unset($inputs);
        $t = $trues; unset($trues);

        $K = $this->backend;
        $totalLoss = 0.0;
        $totalAccuracy = 0.0;
        if(!($callbacks instanceof CallbackList)) {
            $callbacks = new CallbackList($this,$callbacks);
        }
        if($verbose>=1) {
            $startTime = time();
        }
        if($x instanceof NDArray||is_array($x)) {
            $dataset = new NDArrayDataset(
                $K->localMatrixOperator(),$x, tests: $t, batch_size: $batch_size);
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
            if(!is_array($inputs)) {
                $inputs = $K->array($inputs);
            } else {
                $newInputs = [];
                foreach ($inputs as $value) {
                    $newInputs[] = $K->array($value);
                }
                $inputs = $newInputs;
                unset($newInputs);
            }
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

    public function predict(
        $inputs,
        array|object $callbacks=null,
        ...$options
    ) : NDArray
    {
        //if(!$this->built) {
        //    throw new LogicException('Not yet built');
        //}
        $callbacks = $callbacks ?? null;

        if(!($callbacks instanceof CallbackList)) {
            $callbacks = new CallbackList($this,$callbacks);
        }
        if(!is_array($inputs)) {
            $inputs = $this->backend->array($inputs);
        } else {
            $newInputs = [];
            foreach ($inputs as $value) {
                $newInputs[] = $this->backend->array($value);
            }
            $inputs = $newInputs;
            unset($newInputs);
        }
        $callbacks->onPredictBegin();
        $outputs = $this->predictStep($inputs,$options);
        $callbacks->onPredictEnd();
        $outputs = $this->backend->ndarray($outputs);
        return $outputs;
    }

    /**
    *  @return int
    */
    public function generation() : int
    {
        return $this->generation;
    }
    /**
    *  @return array<Variable>
    */
    public function inputs()
    {
        return $this->inputsVariables;
    }

    /**
    *  @return array<Variable>
    */
    public function outputs()
    {
        return $this->outputsVariables;
    }

    protected function getModelGraph()
    {
        if(isset($this->graph['model'])) {
            return $this->graph['model'];
        }
        $model = $this;
        $func = function($x,$training,$t) use ($model) {
            return $model->forward($x,$training,$t);
        };
        //$options = ['alternateCreator'=>$this];
        //[$weights,$grads] = $this->initWeights();
        //if(count($weights)) {
        //    $options['weights'] = $weights;
        //    $options['grads'] = $grads;
        //}
        $this->graph['model'] = $this->builder->gradient->function(
            $func,alternateCreator:$this);
        return $this->graph['model'];
    }

    public function shapeInspection() : bool
    {
        return $this->shapeInspection;
    }

    public function setShapeInspection(bool $enable)
    {
        if($this->shapeInspection==$enable)
            return;
        foreach ($this->submodules() as $module) {
            $module->setShapeInspection($enable);
        }
        $this->shapeInspection = $enable;
    }

    public function submodules() : array
    {
        $modules = [];
        foreach (get_object_vars($this) as $func) {
            if($func instanceof Module) {
                $modules[] = $func;
            }
        }
        return $modules;
    }

    public function layers() : array
    {
        $layers = [];
        $modules = $this->submodules();

        while($mdl = array_shift($modules)) {
            if($mdl instanceof Layer) {
                $layers[] = $mdl;
            } else {
                $modules = array_merge($mdl->submodules(),$modules);
            }
        }
        return $layers;
    }

    public function variables() : array
    {
        $variables = [];
        foreach ($this->submodules() as $module) {
            $variables = array_merge($variables,$module->variables());
        }
        foreach(get_object_vars($this) as $var) {
            if($var instanceof Variable) {
                $variables[] = $var;
            }
        }

        return $variables;
    }

    public function trainableVariables() : array
    {
        return array_filter($this->variables(),fn($v)=>$v->isTrainable());
    }

    public function reverseSyncWeightVariables() : void
    {
    }

    public function __invoke(...$inputs)
    {
        $outputs = $this->forward(...$inputs);
        return $outputs;
    }

    public function forward(...$inputs)
    {
        $outputs = $this->call(...$inputs);
        $this->built = true;
        return $outputs;
    }

    public function _rawCall($inputs)
    {
        return $this->graph['model']->_rawCall($inputs);
    }

    public function backward(array $dOutputs, ArrayAccess $grads=null, array $oidsToCollect=null) : array
    {
        return $this->graph['model']->backward($dOutputs, $grads, $oidsToCollect);
    }

    protected function trainStep($inputs, $trues)
    {
        $K = $this->backend;
        $nn = $this->builder;
        $g = $nn->gradient();
        if(!is_array($inputs)) {
            $inputs = [$inputs];
        }
        $inputs = array_map(fn($x)=>$g->Variable($x),$inputs);
        $inputs[] = $g->Variable(true);
        $inputs[] = $g->Variable($trues);
        $trues = $this->trueValuesFilter($trues);
        $model = $this->getModelGraph();
        $lossfunc = $this->lossFunction;
        [$loss,$preds] = $nn->with($tape=$g->GradientTape(),
            function() use ($K,$model,$lossfunc,$inputs,$trues) {
                $predicts = $model(...$inputs);
                return [$lossfunc($trues,$predicts),$predicts];
            }
        );
        $lossValue = $K->scalar($loss->value());
        if(is_nan($lossValue)) {
            throw new UnexpectedValueException("loss is unexpected value");
        }
        $params = $this->trainableVariables();
        $gradients = $tape->gradient($loss, $params);
        $this->optimizer->update($params, $gradients);

        if(in_array('accuracy',$this->metrics)) {
            //$preds = $this->forwardLastlayer($preds);
            $accuracy = $this->lossFunction->accuracy($trues,$preds->value());
        } else {
            $accuracy = 0;
        }
        return [$lossValue,$accuracy];
    }

    protected function evaluateStep($inputs,$trues)
    {
        $nn = $this->builder;
        $K = $nn->backend();
        $g = $nn->gradient();
        if(!is_array($inputs)) {
            $inputs = [$inputs];
        }
        $inputs = array_map(fn($x)=>$g->Variable($x),$inputs);
        $inputs[] = $g->Variable(false); // training
        $inputs[] = $g->Variable($trues);

        $trues = $this->trueValuesFilter($trues);
        $model = $this->getModelGraph();
        $lossfunc = $this->lossFunction;
        $predicts = $model(...$inputs);
        $loss = $lossfunc($trues,$predicts);
        $loss = $K->scalar($loss->value());
        $accuracy = $this->lossFunction->accuracy($trues,$predicts->value());
        return [$loss,$accuracy];
    }

    protected function predictStep($inputs,$options)
    {
        $nn = $this->builder;
        $g = $nn->gradient();
        if(!is_array($inputs)) {
            $inputs = [$inputs];
        }
        $inputs = array_map(fn($x)=>$g->Variable($x),$inputs);
        $inputs[] = $g->Variable(false); // training
        $inputs[] = $g->Variable(false); // trues
        $model = $this->getModelGraph();
        $predicts = $model(...$inputs);
        return $predicts->value();
    }

    public function parameterVariables() : array
    {
        return [];
    }

    public function build(...$inputShapes) : void
    {
        if($this->built) {
            return;
        }
        $K = $this->backend;
        $inputs = [];
        foreach($inputShapes as $inputShape) {
            if(is_array($inputShape)) {
                $inputs[] = $K->zeros($inputShape);
            } else {
                $inputs[] = $inputShape;
            }
        }
        $this->forward(...$inputs);
    }

    public function summary()
    {
        if(!$this->built) {
            throw new LogicException('You need to build the model before you can see the summary.');
        }
        $this->display(
            str_pad('Layer(type)',29).
            str_pad('Output Shape',27).
            str_pad('Param #',10).
            "\n");
        $this->display(str_repeat('=',29+27+10)."\n");
        $totalParams = 0;
        foreach ($this->layers() as $layer) {
            $type = $this->basename($layer);
            $this->display(substr(str_pad($layer->getName().'('.$type.')',29),0,29));
            $outputShape = $layer->outputShape();
            if(count($outputShape)>0 && is_array($outputShape[0])) {
                $outputShape = $outputShape[0];
            }
            $this->display(str_pad('('.implode(',',$outputShape).')',27));
            $nump = 0;
            foreach($layer->getParams() as $p) {
                $nump += $p->size();
            }
            $this->display(str_pad($nump,10));
            $this->display("\n");
            $totalParams += $nump;
        }
        $params = $this->parameterVariables();
        if(count($params)) {
            $this->display(str_repeat('=',29+27+10)."\n");
            $this->display(
                str_pad('Weights',29).
                str_pad('Shape',27).
                str_pad('Param #',10).
                "\n");
            $this->display(str_repeat('=',29+27+10)."\n");
            foreach($params as $param) {
                $name = $param->name();
                if(!$name) {
                    $name = 'No name';
                }
                $this->display(str_pad($name,29));
                $this->display(str_pad('('.implode(',',$param->shape()).')',27));
                $nump = $param->size();
                $this->display(str_pad($nump,10));
                $this->display("\n");
                $totalParams += $nump;
            }
        }
        $this->display(str_repeat('=',29+27+10)."\n");
        $this->display('Total params: '.$totalParams."\n");
    }

    public function saveWeights(&$modelWeights,$portable=null) : void
    {
        $K = $this->backend;
        $modelWeights['weights'] = $modelWeights['weights'] ?? [];
        foreach($this->variables() as $idx => $weights) {
            $param = $weights->value();
            $param=$K->ndarray($param);
            if($portable)
                $param = $this->converPortableSaveMode($param);
            $modelWeights['weights'][$idx] = serialize($param);
        }
        $optimizerWeights = $this->optimizer()->getWeights();
        $modelWeights['optimizerNumWeight'] = count($optimizerWeights);
        $modelWeights['optimizer'] = $modelWeights['optimizer'] ?? [];
        foreach ($optimizerWeights as $idx => $weights) {
            $weights=$K->ndarray($weights);
            $modelWeights['optimizer'][$idx] = serialize($weights);
        }
    }
    
    public function loadWeights($modelWeights) : void
    {
        $K = $this->backend;
        $nn = $this->builder;
        $g = $nn->gradient();
        $model = $this;
        $lossfunc = $this->lossFunction;
        foreach($this->variables() as $idx => $weights) {
            $data = unserialize($modelWeights['weights'][$idx]);
            $data = $K->array($data);
            $weights->assign($K->copy($data));
        }
        $stack = [$this];
        while($module=array_pop($stack)) {
            $module->reverseSyncWeightVariables();
            foreach($module->submodules() as $m) {
                array_push($stack,$m);
            }
        }
        $optimizer = $this->optimizer();
        //$optimizer->build($this->params());
        $optimizerNumWeight = $modelWeights['optimizerNumWeight'];
        $params = [];
        for($idx=0;$idx<$optimizerNumWeight;$idx++) {
            $data = unserialize($modelWeights['optimizer'][$idx]);
            $data = $K->array($data);
            //$K->copy($data,$weights);
            $params[] = $data;
        }
        $optimizer->loadWeights($params);
    }

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

    public function __clone()
    {
        $props = get_object_vars($this);
        foreach ($props as $name => $value) {
            if($value instanceof Layer) {
                $this->$name = clone $value;
            } elseif($value instanceof Model) {
                $this->$name = clone $value;
            }
        }
        $this->built = false;
        //$this->params = [];
        //$this->grads = [];
        $this->pipeline = [];
        $this->layers = [];
        $this->outputsVariables = null;
        $this->optimizer = null;
        $this->lossFunction = null;
        $this->metrics = null;
    }
}
