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
        ?int $axis=null,
        ?float $epsilon=null,
        ?bool $center=null,
        ?bool $scale=null,
        string|callable|null $beta_initializer=null,
        string|callable|null $gamma_initializer=null,
        ?string $name=null,
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
        $this->allocateWeights(['beta','gamma']);
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

    protected function call(NDArray $inputs, ?bool $training=null) : NDArray
    {
        $K = $this->backend;
        //if($training===null) {
        //    throw new InvalidArgumentException("training option must be true or false.");
        //}
        $container = $this->container();
        // (batch,heads...,feature) => (batch*heads,feature)
        $inputs = $this->transformShape($inputs);

        // normalization
        // xn = (x - mean(x)) / sqrt(mean( (x - mean(x))**2 ) + eps)
        //

        // mean = mean(x)
        // center = x - mean(x)
        $mean = $K->mean($inputs,axis:-1);                          // (batch*heads)
        $center_x = $K->sub($inputs, $mean, trans:true);            // (batch*heads,feature)

        // variance = mean(square(x - mean), axis=-1)
        $variance = $K->mean($K->square($center_x), axis:-1);       // (batch*heads)

        // std = sqrt(variance+eps)
        // normalized_x = x-mean(x) / std
        $std = $K->sqrt($K->increment($variance, $this->epsilon));  // (batch*heads)
        $norm_x = $K->div($center_x, $std, trans:true);             // (batch*heads,feature)

        $container->norm_x = $norm_x;   // (batch*head,feature)
        $container->std = $std;         // (batch*heads)

        if($this->gamma) {
            $outputs = $K->mul($this->gamma, $norm_x);
        } else {
            $outputs = $norm_x;
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
        $dOutputs = $this->transformShape($dOutputs);
        $container = $this->container();
        $norm_x = $container->norm_x;           // (batch*head,feature)
        $std = $container->std;                 // (batch*heads)

        $tmp = $dOutputs->shape();
        $feature_dim = array_pop($tmp);

        // d_scaled_x = dOutputs                // (batch*head,feature)
        // d_norm_x = d_scaled_x * gamma        // (batch*head,feature)
        if($this->dBeta) {
            $dbeta = $K->sum($dOutputs,axis:0,output:$this->dBeta);
        }
        if($this->dGamma) {
            $dgamma = $K->sum($K->mul($norm_x, $dOutputs), axis:0, output:$this->dGamma);
            $d_norm_x = $K->mul($this->gamma, $dOutputs);    // (batch*head,feature)
        } else {
            $d_norm_x = $dOutputs;                           // (batch*head,feature)
        }
        if($std===null) {
            throw new LogicException('not initialized for training');
        }
        // d_center_x = d_normalized_x / std
        $d_center_x = $K->div($d_norm_x, $std, trans:true);             // (batch*head,feature)
        // d_mean = sum(d_center_x)
        $d_mean = $K->sum($d_center_x, axis:-1);                        // (batch*head)
        // d_std = sum(normalized_x*d_center_x)
        $d_std = $K->sum($K->mul($norm_x,$d_center_x), axis:-1);        // (batch*head)
        // d_x = d_center_x - (d_mean + normalized_x*d_std) / feature_dim_f
        $dInputs = $K->sub(                 // (batch*head,feature)
            $d_center_x,                    // (batch*head,feature)
            $K->scale(1/$feature_dim,       // (batch*head,feature)
                $K->add(                    // (batch*head,feature)
                    $K->mul(                // (batch*head,feature)
                        $norm_x,            // (batch*head,feature)
                        $d_std,             // (batch*head)
                        trans:true,
                    ),
                    $d_mean,                // (batch*head)
                    trans:true,
                )
            )
        );

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

        $this->allocateWeights(array_map(fn($weight)=>$weight->name(),$this->weights));
        if($this->assignedWeights) {
            $this->syncWeightVariables();
        }
    }
}
