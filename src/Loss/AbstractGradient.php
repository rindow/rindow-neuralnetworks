<?php
namespace Rindow\NeuralNetworks\Loss;

use Interop\Polite\Math\Matrix\NDArray;
//use Rindow\NeuralNetworks\Activation\Activation;
use Rindow\NeuralNetworks\Gradient\Core\Variable;
use Rindow\NeuralNetworks\Gradient\Core\Undetermined;
use Rindow\NeuralNetworks\Gradient\Core\GradientTape;
use Rindow\NeuralNetworks\Gradient\Core\GradientUtils;
use Rindow\NeuralNetworks\Support\GenericUtils;
use InvalidArgumentException;
use DomainException;

abstract class AbstractGradient //implements Loss
{
    use GradientUtils;
    protected $generation;
    protected $inputsVariables;
    protected $outputsVariables;

    /*
    *  dinamic step interfaces
    */
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

    /**
    *  @param array<Variable>  $inputs
    *       inputs
    *  @return array<Variable>
    *       outputs
    */
    public function __invoke($trues, $predicts)
    {
        $K = $this->backend;
        $inputs = $predicts;
        //if($inputs instanceof Undetermined) {
        //    $outputs = new Undetermined();
        //} else {
        //    $inputValues = $inputs->value();
        //    $outValue = $this->forward($trues,$inputValues);
        //    $outputs = new Variable($this->backend,
        //        $K->array($outValue,$inputValues->dtype()));
        //}
        //if(GradientTape::$autoBackProp) {
        //    $this->generation = $inputs->generation();
        //    $outputs->setCreator($this);
        //    $this->inputsVariables = [$inputs];
        //    $this->outputsVariables = [$outputs->reference()];
        //}
        //return $outputs;

        if($inputs instanceof Undetermined) {
            $outputs = null;
        } else {
            $inputValues = $inputs->value();
            $outputs = $this->forward($trues,$inputValues);
            $outputs = $K->array($outputs,$inputValues->dtype());
        }
        $outputs = $this->postGradientProcess(
            $this->backend, [$inputs], [$outputs]);
        return $outputs[0];
    }
}
