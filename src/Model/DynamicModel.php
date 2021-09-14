<?php
namespace Rindow\NeuralNetworks\Model;

use InvalidArgumentException;
use UnexpectedValueException;
use LogicException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Layer\LayerBase;
use Rindow\NeuralNetworks\Gradient\Core\Variable;
use Rindow\NeuralNetworks\Gradient\Core\Undetermined;
use Rindow\NeuralNetworks\Gradient\Core\UndeterminedNDArray;

abstract class DynamicModel extends AbstractModel
{
    protected $weightVariables = [];
    protected $trainableVariables;
    protected $generation;
    protected $inputsVariables;
    protected $outputsVariables;

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

    /*
    *  dinamic step interfaces
    */
    /**
    *  @param LayerBase|Variable  $weights
    *       inputs
    *  @return LayerBase|Variable
    *       outputs
    */
    public function add($weights)
    {
        if($weights instanceof LayerBase||
            $weights instanceof DynamicModel) {
            $this->layers[] = $weights;
            $this->weightVariables[] = $weights;
        } elseif($weights instanceof Variable) {
            $this->weightVariables[] = $weights;
        } else {
            throw new InvalidArgumentException('weights must be Variable or Layer');
        }
        return $weights;
    }

    public function trainableVariables()
    {
        return $this->weights();
    }

    public function weights()
    {
        if($this->trainableVariables) {
            return $this->trainableVariables;
        }
        $this->trainableVariables = [];
        foreach($this->weightVariables as $weights) {
            if(($weights instanceof LayerBase)||($weights instanceof DynamicModel)) {
                $this->trainableVariables = array_merge($this->trainableVariables,$weights->weights());
            } else {
                $this->trainableVariables = array_merge($this->trainableVariables,[$weights]);
            }
        }
        return $this->trainableVariables;
    }

    public function parameterVariables() : array
    {
        $weightVariables = $this->trainableVariables();
        $params = [];
        foreach ($weightVariables as $weights) {
            if($weights instanceof Variable) {
                $params[] = $weights;
            }
        }
        return $params;
    }


    /**
    *  @param array<Variable>  $inputs
    *       inputs
    *  @return array<Variable>
    *       outputs
    */
    public function __invoke(...$inputs)
    {
        $outputs = $this->call(...$inputs);
        return $outputs;
    }

    /*
    *  dinamic step interfaces
    */
    /**
    *  @param array<Variable>  $inputs
    *       inputs
    *  @return array<Variable>
    *       outputs
    */
    //protected function call(...$inputs) : array
    //{
    //    throw new LogicException('"call" is not implemented');
    //}

    protected function buildLayers(array $options=null) : void
    {
    //    $model = $this;
    //    $model($x,true)
    }

    protected function trainStep($inputs, $trues)
    {
        $K = $this->backend;
        $nn = $this->builder;
        $g = $nn->gradient();
        $x = $g->Variable($inputs);
        $t = $g->Variable($trues);
        $trues = $this->trueValuesFilter($trues);
        $model = $this;
        $lossfunc = $this->lossFunction;
        [$loss,$preds] = $nn->with($tape=$g->GradientTape(),
            function() use ($K,$model,$lossfunc,$x,$t,$trues) {
                $predicts = $model($x,true,$t);
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
        $x = $g->Variable($inputs);
        $t = $g->Variable($trues);
        $trues = $this->trueValuesFilter($trues);
        $model = $this;
        $lossfunc = $this->lossFunction;
        $predicts = $model($x,false,$t);
        $loss = $lossfunc($trues,$predicts);
        $loss = $K->scalar($loss->value());
        $accuracy = $this->lossFunction->accuracy($trues,$predicts->value());
        return [$loss,$accuracy];
    }

    protected function predictStep($inputs,$options)
    {
        $nn = $this->builder;
        $g = $nn->gradient();
        $x = $g->Variable($inputs);
        $model = $this;
        $predicts = $model($x,false,null);
        return $predicts->value();
    }

    public function saveWeights(&$modelWeights,$portable=null) : void
    {
        $K = $this->backend;
        if(!isset($modelWeights['weights']))
            $modelWeights['weights'] = [];
        foreach($this->trainableVariables() as $idx => $weights) {
            $param = $weights->value();
            $param=$K->ndarray($param);
            if($portable)
                $param = $this->converPortableSaveMode($param);
            $modelWeights['weights'][$idx] = serialize($param);
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
        $nn = $this->builder;
        $g = $nn->gradient();
        $x = new Undetermined();
        $t = new Undetermined();
        $model = $this;
        $lossfunc = $this->lossFunction;
        [$loss,$preds] = $nn->with($tape=$g->GradientTape(),
            function() use ($K,$model,$lossfunc,$x,$t) {
                $predicts = $model($x,true,$t);
                return [$lossfunc($t,$predicts),$predicts];
            }
        );

        foreach($this->trainableVariables() as $idx => $weights) {
            $param = $weights->value();
            $data = unserialize($modelWeights['weights'][$idx]);
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

    public function summary()
    {
        throw new LogicException('"Unsupported function');
    }

    //public function save($filepath,$portable=null) : void
    //{
    //    throw new LogicException('"Unsupported function');
    //}
}
