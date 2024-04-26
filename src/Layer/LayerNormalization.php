<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use LogicException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class LayerNormalization extends AbstractNormalization
{
    public function __construct(
        object $backend,
        int $axis=null,
        float $epsilon=null,
        bool $center=null,
        bool $scale=null,
        string|callable $beta_initializer=null,
        string|callable $gamma_initializer=null,
        string $name=null,
    )
    {
        parent::__construct(
            $backend,
            $axis,
            $epsilon,
            $center,
            $scale,
            $beta_initializer,
            $gamma_initializer,
        );
        // defaults
        $name = $name ?? null;

        $this->initName($name,'layernormalization');
        $this->allocateWeights(2);
    }

    protected function buildNoTrainingMode(array $kernelShape) : void
    {
    }

    public function reverseSyncWeightVariables() : void
    {
        $this->beta = $this->weights[0]->value();
        $this->gamma = $this->weights[1]->value();
    }

    public function getConfig() : array
    {
        return [
            'options' => [
                'axis'=>$this->axis,
                'epsilon'=>$this->epsilon,
                'beta_initializer'=>$this->betaInitializerName,
                'gamma_initializer'=>$this->gammaInitializerName,
            ]
        ];
    }

    public function getParams() : array
    {
        return [$this->beta,$this->gamma];
    }

    public function getGrads() : array
    {
        return [$this->dBeta,$this->dGamma];
    }

    protected function call(NDArray $inputs, bool $training=null) : NDArray
    {
        $K = $this->backend;
        if($training===null) {
            throw new InvalidArgumentException("training option must be true or false.");
        }
        $container = $this->container();
        // (batch,heads...,feature) => (batch*heads,feature)
        $inputs = $this->transformShape($inputs);

        // normalization
        // xn = (x - mean(x)) / sqrt(mean( (x - mean(x))**2 ) + eps)
        $shape = $inputs->shape();
        $size = array_pop($shape);
        $mu = $K->mean($inputs,axis:-1);
        $muEx = $K->expandDims($mu,axis:-1);
        $muEx = $K->repeat($muEx,$size,axis:-1);
        $muEx = $K->squeeze($muEx,axis:-1);
        $xc = $K->sub($inputs, $muEx);

        $v = $K->mean($K->square($xc), axis:-1);
        $vEx = $K->expandDims($v,axis:-1);
        $vEx = $K->repeat($vEx,$size,axis:-1);
        $vEx = $K->squeeze($vEx,axis:-1);

        $std = $K->sqrt($K->increment($vEx, $this->epsilon));
        $xn = $K->div($xc, $std);

        $container->xc = $xc;
        $container->xn = $xn;
        $container->std = $std;

        if($this->gamma) {
            $outputs = $K->mul($this->gamma, $xn);
        } else {
            $outputs = $xn;
        }
        if($this->beta) {
            $outputs = $K->add($outputs, $this->beta);
        }

        $outputs = $this->untransformShape($outputs);
        return $outputs;
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        $container = $this->container();
        $dOutputs = $this->transformShape($dOutputs);
        $numItems = $dOutputs->shape()[0];

        if($this->dBeta) {
            $dbeta = $K->sum($dOutputs,axis:0,output:$this->dBeta);
            //$K->copy($dbeta,$this->dBeta);
        }
        if($this->dGamma) {
            $dgamma = $K->sum($K->mul($container->xn, $dOutputs), axis:0, output:$this->dGamma);
            //$K->copy($dgamma,$this->dGamma);
            $dxn = $K->mul($this->gamma, $dOutputs);
        } else {
            $dxn = $dOutputs;
        }
        if($container->std===null) {
            throw new LogicException('not initialized for training');
        }
        $dxc = $K->div($dxn, $container->std);
        $shape = $dxn->shape();
        $size = array_pop($shape);
        $dstd = $K->scale(-1.0, $K->sum(
            $K->div($K->mul($dxn, $container->xc), $K->mul($container->std, $container->std)),
            axis:-1));
        $dstd = $K->expandDims($dstd,$axis=-1);
        $dstd = $K->repeat($dstd,$size,axis:-1);
        $dstd = $K->squeeze($dstd,axis:-1);

        $dvar = $K->div($K->scale(0.5, $dstd), $container->std);
        $K->update_add($dxc,
            $K->scale(2.0/$numItems, $K->mul($container->xc, $dvar)));
        $dmu = $K->sum($dxc, axis:-1);
        $dmu = $K->expandDims($dmu,$axis=-1);
        $dmu = $K->repeat($dmu,$size,axis:-1);
        $dmu = $K->squeeze($dmu,axis:-1);

        $dInputs = $K->sub($dxc, $K->scale(1/$numItems,$dmu));

        $dInputs = $this->untransformShape($dInputs);
        return $dInputs;
    }

    public function __clone()
    {
        if(isset($this->gamma)) {
            $this->gamma = clone $this->gamma;
        }
        if(isset($this->beta)) {
            $this->beta = clone $this->beta;
        }
        if(isset($this->dGamma)) {
            $this->dGamma = clone $this->dGamma;
        }
        if(isset($this->dBeta)) {
            $this->dBeta = clone $this->dBeta;
        }

        $this->allocateWeights(2);
        if($this->assignedWeights) {
            $this->syncWeightVariables();
        }
    }
}
