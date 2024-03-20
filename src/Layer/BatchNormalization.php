<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class BatchNormalization extends AbstractNormalization
{
    public function __construct(
        object $backend,
        int $axis=null,
        float $momentum=null,
        float $epsilon=null,
        bool $center=null,
        bool $scale=null,
        string|callable $beta_initializer=null,
        string|callable $gamma_initializer=null,
        string|callable $moving_mean_initializer=null,
        string|callable $moving_variance_initializer=null,
        string $name=null,
    )
    {
        // defaults
        $axis = $axis ?? -1;
        $momentum = $momentum ?? 0.99;
        $epsilon = $epsilon ?? 0.001;
        $center = $center ?? true;
        $scale = $scale ?? true;
        $beta_initializer = $beta_initializer ?? 'zeros';
        $gamma_initializer = $gamma_initializer ?? 'ones';
        $moving_mean_initializer = $moving_mean_initializer ?? 'zeros';
        $moving_variance_initializer = $moving_variance_initializer ?? 'ones';
        $name = $name ?? null;

        $this->backend = $K = $backend;
        $this->axis = $axis;
        $this->momentum = $momentum;
        $this->epsilon = $epsilon;
        $this->center = $center;
        $this->scale = $scale;
        $this->betaInitializerName  = $beta_initializer;
        $this->gammaInitializerName = $gamma_initializer;
        $this->movingMeanInitializerName  = $moving_mean_initializer;
        $this->movingVarianceInitializerName = $moving_variance_initializer;
        $this->betaInitializer  = $K->getInitializer($beta_initializer);
        $this->gammaInitializer = $K->getInitializer($gamma_initializer);
        $this->movingMeanInitializer  = $K->getInitializer($moving_mean_initializer);
        $this->movingVarianceInitializer = $K->getInitializer($moving_variance_initializer);
        $this->initName($name,'batchnormalization');
        $this->allocateWeights(2,$nonTrainables=2);
        $this->callOptions['training'] = true;
    }

    protected function buildNoTrainingMode(array $kernelShape) : void
    {
        $movingMeanInitializer = $this->movingMeanInitializer;
        $movingVarianceInitializer = $this->movingVarianceInitializer;

        if($this->movingMean==null) {
            $this->movingMean = $movingMeanInitializer($kernelShape);
            $this->movingVariance = $movingVarianceInitializer($kernelShape);
        }
    }

    public function reverseSyncWeightVariables() : void
    {
        $this->beta = $this->weights[0]->value();
        $this->gamma = $this->weights[1]->value();
        $this->movingMean = $this->weights[2]->value();
        $this->movingVariance = $this->weights[3]->value();
    }

    public function getConfig() : array
    {
        return [
            'options' => [
                'axis'=>$this->axis,
                'momentum'=>$this->momentum,
                'epsilon'=>$this->epsilon,
                'beta_initializer'=>$this->betaInitializerName,
                'gamma_initializer'=>$this->gammaInitializerName,
                'moving_mean_initializer'=>$this->movingMeanInitializerName,
                'moving_variance_initializer'=>$this->movingVarianceInitializerName,
            ]
        ];
    }

    public function getParams() : array
    {
        return [$this->beta,$this->gamma,$this->movingMean,$this->movingVariance];
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
        if($training) {
            // xn = (x - mean(x)) / sqrt(mean( (x - mean(x))**2 ) + eps)
            $mu = $K->mean($inputs,axis:0);
            $xc = $K->sub($inputs, $mu);
            $v = $K->mean($K->square($xc), axis:0);
            $std = $K->sqrt($K->increment($v, $this->epsilon));
            $xn = $K->div($xc, $std);

            $container->xc = $xc;
            $container->xn = $xn;
            $container->std = $std;
            // stateful variable
            // movingMean = movingMean*momentum + mu*(1-momentum)
            $K->update_scale($this->movingMean,$this->momentum);
            $K->update_add($this->movingMean,$K->scale(1-$this->momentum, $mu));
            // movingVariance = movingVariance*momentum + v*(1-momentum)
            $K->update_scale($this->movingVariance,$this->momentum);
            $K->update_add($this->movingVariance,$K->scale(1-$this->momentum, $v));

            //$this->movingMean =     $K->add($K->scale($this->momentum,   $this->movingMean),
            //                                $K->scale(1-$this->momentum, $mu));
            //$this->movingVariance = $K->add($K->scale($this->momentum,   $this->movingVariance),
            //                                $K->scale(1-$this->momentum, $v));
        } else { // not training
            // xn = (x - movingMean) / sqrt(movingVariance + eps)
            $xc = $K->sub($inputs, $this->movingMean);
            $xn = $K->div($xc, ($K->sqrt($K->increment($this->movingVariance, $this->epsilon))));
            $container->std = null;
        }

        if($this->gamma) {
            $outputs = $K->mul($this->gamma, $xn);
        } else {
            $outputs = $xn;
        }
        if($this->beta) {
            $outputs = $K->add($outputs, $this->beta);
        }

        // (batch*heads,feature) => (batch,heads...,feature)
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
        if($container->std===null)
            throw new LogicException('not initialized for training');
        $dxc = $K->div($dxn, $container->std);
        $dstd = $K->scale(-1.0, $K->sum(
            $K->div($K->mul($dxn, $container->xc), $K->mul($container->std, $container->std)),
            axis:0));
        $dvar = $K->div($K->scale(0.5, $dstd), $container->std);
        $K->update_add($dxc,
            $K->scale(2.0/$numItems, $K->mul($container->xc, $dvar)));
        $dmu = $K->sum($dxc, axis:0);
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
        if(isset($this->movingMean)) {
            $this->movingMean = clone $this->movingMean;
        }
        if(isset($this->movingVariance)) {
            $this->movingVariance = clone $this->movingVariance;
        }

        if(isset($this->dGamma)) {
            $this->dGamma = clone $this->dGamma;
        }
        if(isset($this->dBeta)) {
            $this->dBeta = clone $this->dBeta;
        }

        $this->allocateWeights(2,$nonTrainables=2);
        $this->syncWeightVariables();
    }
}
