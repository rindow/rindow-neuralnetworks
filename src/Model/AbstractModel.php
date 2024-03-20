<?php
namespace Rindow\NeuralNetworks\Model;

use InvalidArgumentException;
use UnexpectedValueException;
use LogicException;
use ReflectionClass;
use ArrayAccess;
use Rindow\NeuralNetworks\Builder\Builder;
use Rindow\NeuralNetworks\Optimizer\Optimizer;
use Rindow\NeuralNetworks\Layer\Layer;
use Rindow\NeuralNetworks\Activation\Softmax;
use Rindow\NeuralNetworks\Activation\Sigmoid;
use Rindow\NeuralNetworks\Loss\Loss;
use Rindow\NeuralNetworks\Loss\SparseCategoricalCrossEntropy;
use Rindow\NeuralNetworks\Loss\CategoricalCrossEntropy;
use Rindow\NeuralNetworks\Loss\BinaryCrossEntropy;
use Rindow\NeuralNetworks\Metric\Metric;
use Rindow\NeuralNetworks\Metric\MetricCatalog;
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
    protected $metrics = [];
    protected $lossFunction;
    protected $accuracyFunction;
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
    protected $callOptions;

    public function __construct(Builder $builder,$hda=null)
    {
        $this->builder = $builder;
        $this->backend = $builder->backend();
        if($hda===null) {
            $this->hda = $builder->utils()->HDA();
        } else {
            $this->hda = $hda;
        }
        $refClass = new ReflectionClass($this);
        $refParams = $refClass->getMethod('call')->getParameters();
        $this->callOptions = [];
        foreach($refParams as $param) {
            $this->callOptions[$param->name] = true;
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

    // public function accuracyFunction()
    // {
    //     return $this->accuracyFunction;
    // }

    public function optimizer()
    {
        return $this->optimizer;
    }

    protected function basename($object) : string
    {
        $classname = get_class($object);
        return substr($classname,strrpos($classname,'\\')+1);
    }

    public function isAwareOf(string $name) : bool
    {
        return isset($this->callOptions[$name]);
    }

    protected function setAwareOf(string $name) : bool
    {
        $this->callOptions[$name] = true;
    }

    protected function resolveOptimizer(mixed $optimizer) : mixed
    {
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
        return $optimizer;
    }

    protected function resolveLossFunction(mixed $loss) : mixed
    {
        if(is_string($loss)) {
            $loss = strtolower($loss);
        }
        if($loss=='sparsecategoricalcrossentropy'||
            $loss=='sparse_categorical_crossentropy') {
            $loss = $this->builder->losses()->SparseCategoricalCrossEntropy();
        }
        if(is_callable($loss)) {
            return $loss;
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
        return $loss;
    }

    protected function resolveMetricFunctions(?array $metrics) : array
    {
        if(empty($metric)) {
            $metric = [];
        }
        $newMetrics = [];
        foreach($metrics as $idx => $metric) {
            if(is_string($metric)) {
                $name = $metric;
                if($metric == 'loss') {
                    $metricObject = $this->builder->metrics()->ScalarMetric(name:'loss');
                } else {
                    if($metric=='accuracy') {
                        $metric = $this->lossFunction->accuracyMetric($metric); // string name
                    }
                    $metricObject = MetricCatalog::factory($this->backend,$metric);
                }
            } elseif($metric instanceof Metric) {
                $metricObject = $metric;
                $name = $metric->name();
            } elseif(is_callable($metric)) {
                $metricObject = $this->builder->metrics()->GenericMetric($metric);
                $name = $metricObject->name();
            } else {
                if(is_object($metric)) {
                    $name = get_class($metric);
                } else {
                    $name = gettype($metric);
                }
                throw new InvalidArgumentException('Invarid metric type:'.$name);
            }
            if(!is_int($idx)) {
                $name = $idx;
            }
            $newMetrics[$name] = $metricObject;
        }
        return $newMetrics;
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

        $this->optimizer = $this->resolveOptimizer($optimizer);

        $this->lossFunction = $this->resolveLossFunction($loss);
        if($this->lossFunction instanceof Loss) {
            $this->accuracyFunction = [$this->lossFunction,'accuracy'];
        }
        // resolve metrics
        $metrics = $this->resolveMetricFunctions($metrics);
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
        if($this->callOptions===null) {
            throw new LogicException('Not yet initialized by constructor.');
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
            if(is_array($val_inputs)) {
                $val_inputs = new NDArrayDataset(
                    $K->localMatrixOperator(),
                    $val_inputs,
                    batch_size: $batch_size,
                    shuffle: $shuffle,
                    filter: $filter,
                    tests: $val_test,
                );
                $val_test = null;
            }
        } elseif($validation_data instanceof Dataset) {
            $val_inputs = $validation_data;
            $val_test = null;
        } else {
            throw new InvalidArgumentException('unsupported dataset type.'.
                ' validation_data must be set of NDArray or instance of Dataset.');
        }
        foreach($this->metrics as $idx => $m) {
            $history[$idx] = [];
        }
        if($val_inputs) {
            foreach($this->metrics as $idx => $m) {
                $history['val_'.$idx] = [];
            }
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
            foreach($this->metrics as $metric) {
                $metric->reset();
            }
            $logs = [];
            $this->trainProcess($dataset,$epoch,$epochs,$startTime,$totalSteps,
                                                    $verbose,$callbacks);
            if($totalSteps==0) {
                $totalSteps = count($dataset);
            }
            if($totalSteps==0) {
                $totalSteps=1;
            }
            foreach($this->metrics as $name => $metric) {
                $value = $metric->result();
                $history[$name][] = $value;
                $logs[$name] = $value;
            }
            if($val_inputs) {
                $evals = $this->evaluate($val_inputs, $val_test,
                    batch_size:$batch_size, verbose:0, callbacks:$callbacks);
 
                foreach($this->metrics as $name => $metric) {
                    $history['val_'.$name][] = $evals[$name];
                    $logs['val_'.$name] = $evals[$name];
                }
            }
            if($verbose>=1) {
                $sec = time() - $startTime;
                $this->console("- {$sec} sec.\n");
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
        $dataset,$epoch,$epochs,$startTime,$totalSteps,$verbose,$callbacks) : void
    {
        $K = $this->backend;
        if($verbose>=1) {
            $this->console('Epoch '.($epoch+1).'/'.$epochs." ");
        }
        if($totalSteps==0) {
            $indicateCount = 1000;
        } else {
            $indicateCount = (int)($totalSteps/25);
        }
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

            $this->trainStep($batchIndex,$inputs, $trues,$callbacks);
            if($this->shapeInspection) {
                $this->setShapeInspection(false);
            }
        }
        if($verbose==1) {
            $this->progressBar($epoch,$epochs,$startTime,
                    $totalSteps,$totalSteps,25);
        } elseif($verbose>1) {
            $this->console(".");
        }
    }

    protected function progressBar($epoch,$epochs,$startTime,$batchIndex,$batchIndexCount,$maxDot)
    {
        $epoch++;
        if($batchIndex==0) {
            $this->console("\rEpoch {$epoch}/{$epochs} ");
            return;
        }
        $elapsed = time() - $startTime;
        if($batchIndexCount) {
            $completion = $batchIndex/$batchIndexCount;
            $estimated = (int)floor($elapsed / $completion);
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
            "] {$elapsed} sec. remaining:{$rem_string}  ");
    }

    protected function trueValuesFilter(NDArray $trues) : NDArray
    {
        return $trues;
    }

    //protected function loss(NDArray $trues,NDArray $preds) : float
    //{
    //    return $this->lossFunction->forward($trues,$preds);
    //}
    //
    //protected function accuracy(NDArray $trues,NDArray $preds) : float
    //{
    //    return $this->lossFunction->accuracy($trues,$preds);
    //}

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
        foreach($this->metrics as $metric) {
            $metric->reset();
        }
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

            $this->evaluateStep($batchIndex,$inputs,$trues,$callbacks);

        }
        $logs = [];
        foreach($this->metrics as $name => $metric) {
            $logs[$name] = $metric->result();
        }
        if($verbose>=1) {
            $sec = time() - $startTime;
            $this->console(' - '.$sec." sec.\n");
            if(array_key_exists('loss',$logs)) {
                $this->console(' loss:'.sprintf('%2.4f',$logs['loss']));
            }
            if(array_key_exists('accuracy',$logs)) {
                $this->console(' accuracy:'.sprintf('%2.4f',$logs['accuracy']));
            }
            $this->console("\n");
        }
        $callbacks->onTestEnd($logs);
        return $logs;
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
        $func = function($x,...$options) use ($model) {
            return $model->forward($x,...$options);
        };
        //$options = ['alternateCreator'=>$this];
        //[$weights,$grads] = $this->initWeights();
        //if(count($weights)) {
        //    $options['weights'] = $weights;
        //    $options['grads'] = $grads;
        //}
        $this->graph['model'] = $this->builder->gradient->Function(
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

    public function parameterVariables() : array
    {
        $variables = [];
        foreach ($this->submodules() as $module) {
            if($module instanceof Model) {
                $variables = array_merge($variables,$module->parameterVariables());
            }
        }
        foreach(get_object_vars($this) as $var) {
            if($var instanceof Variable) {
                if($var->isbackpropagatable()) {
                    $variables[] = $var;
                }
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

    protected function trainStep($batchIndex, $inputs, $trues ,$callbacks) : void
    {
        $K = $this->backend;
        $nn = $this->builder;
        $callbacks->onTrainBatchBegin($batchIndex);

        $g = $nn->gradient();
        if(!is_array($inputs)) {
            $inputs = [$inputs];
        }
        $inputs = array_map(fn($x)=>$g->Variable($x),$inputs);
        if($this->isAwareOf('training')) {
            $inputs['training'] = $g->Variable(true);
        }
        if($this->isAwareOf('trues')) {
            $inputs['trues'] = $g->Variable($trues);
        }
        $trues = $this->trueValuesFilter($trues);
        $trues = $g->Variable($trues);
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

        foreach($this->metrics as $name => $metric) {
            if($name=='loss') {
                $metric->immediateUpdate($lossValue);
            } else {
                $metric->update($trues,$preds->value());
            }
        }
        $callbacks->onTrainBatchEnd($batchIndex,$this->metrics);
    }

    protected function evaluateStep($batchIndex,$inputs,$trues,$callbacks) : void
    {
        $nn = $this->builder;
        $callbacks->onTestBatchBegin($batchIndex);

        $K = $nn->backend();
        $g = $nn->gradient();
        if(!is_array($inputs)) {
            $inputs = [$inputs];
        }
        $inputs = array_map(fn($x)=>$g->Variable($x),$inputs);
        if($this->isAwareOf('training')) {
            $inputs['training'] = $g->Variable(false); // training
        }
        if($this->isAwareOf('trues')) {
            $inputs['trues'] = $g->Variable($trues);
        }

        $trues = $this->trueValuesFilter($trues);
        $model = $this->getModelGraph();
        $lossfunc = $this->lossFunction;
        $predicts = $model(...$inputs);
        $loss = $lossfunc($trues,$predicts);
        $loss = $K->scalar($loss->value());
        foreach($this->metrics as $name => $metric) {
            if($name=='loss') {
                $metric->immediateUpdate($loss);
            } else {
                $metric->update($trues,$predicts->value());
            }
        }
        $callbacks->onTestBatchEnd($batchIndex,$this->metrics);
    }

    protected function predictStep($inputs,$options)
    {
        $nn = $this->builder;
        $g = $nn->gradient();
        if(!is_array($inputs)) {
            $inputs = [$inputs];
        }
        $inputs = array_map(fn($x)=>$g->Variable($x),$inputs);
        if($this->isAwareOf('training')) {
            $inputs['training'] = $g->Variable(false); // training
        }
        if($this->isAwareOf('trues')) {
            $inputs['trues'] = $g->Variable(false); // trues
        }
        $model = $this->getModelGraph();
        $predicts = $model(...$inputs);
        return $predicts->value();
    }

    public function build(...$inputShapes) : void
    {
        if($this->built) {
            return;
        }
        $K = $this->backend;
        $nn = $this->builder;
        $inputs = [];
        foreach($inputShapes as $idx => $inputShape) {
            if(is_array($inputShape)) {
                $inputs[$idx] = $nn->gradient()->Variable($K->zeros($inputShape));
            } else {
                $inputs[$idx] = $inputShape;
            }
        }
        if($this->isAwareOf('training')) {
            $inputs['training'] = false;
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
            $layerName = $layer->getName().'('.$type.')';
            $this->display(substr(str_pad($layerName,29),0,29));
            if(!$layer->isBuilt()) {
                throw new LogicException('the layer is not built: '.$layerName);
            }
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
        $mo = $K->localMatrixOperator();
        $modelWeights['weights'] = $modelWeights['weights'] ?? [];
        foreach($this->variables() as $idx => $weights) {
            $param = $weights->value();
            $param=$K->ndarray($param);
            if($portable)
                $param = $this->converPortableSaveMode($param);
            $modelWeights['weights'][$idx] = $mo->serializeArray($param);
        }
        $optimizerWeights = $this->optimizer()->getWeights();
        $modelWeights['optimizerNumWeight'] = count($optimizerWeights);
        $modelWeights['optimizer'] = $modelWeights['optimizer'] ?? [];
        foreach ($optimizerWeights as $idx => $weights) {
            $weights=$K->ndarray($weights);
            if($portable)
                $weights = $this->converPortableSaveMode($weights);
            $modelWeights['optimizer'][$idx] = $mo->serializeArray($weights);
        }
    }
    
    public function loadWeights($modelWeights) : void
    {
        $K = $this->backend;
        $mo = $K->localMatrixOperator();
        foreach($this->variables() as $idx => $weights) {
            $data = $mo->unserializeArray($modelWeights['weights'][$idx]);
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
            $data = $mo->unserializeArray($modelWeights['optimizer'][$idx]);
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
        $this->outputsVariables = null;
        $this->optimizer = null;
        $this->lossFunction = null;
        $this->accuracyFunction = null;
        $this->metrics = null;
    }
}
