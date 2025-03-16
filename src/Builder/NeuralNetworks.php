<?php
namespace Rindow\NeuralNetworks\Builder;

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend as RindowBlasBackend;
use Rindow\NeuralNetworks\Backend\RindowCLBlast\Backend as RindowCLBlastBackend;
use Rindow\NeuralNetworks\Support\Control\Execute;
use Rindow\NeuralNetworks\Support\Control\Context;
use LogicException;

class NeuralNetworks implements Builder
{
    /** @var array<string,string> $backendClasses */
    protected array $backendClasses = [
        'rindowblas' => RindowBlasBackend::class,
        'rindowclblast' => RindowCLBlastBackend::class,
    ];
    protected object $backend;
    protected object $matrixOperator;
    protected ?object $models=null;
    protected ?object $layers=null;
    protected ?object $losses=null;
    protected ?object $metrics=null;
    protected ?object $optimizers=null;
    protected ?object $networks=null;
    protected ?object $datasets=null;
    protected ?object $data=null;
    protected ?object $utils=null;
    protected ?object $gradient=null;

    public function __construct(?object $matrixOperator=null, ?object $backend=null)
    {
        if($backend==null) {
            if($matrixOperator==null) {
                $matrixOperator = new MatrixOperator();
            }
            $backendname = getenv('RINDOW_NEURALNETWORKS_BACKEND');
            if($backendname) {
                $origName = $backendname;
                $options = explode('::',$backendname);
                $backendname = array_shift($options);
                if(isset($this->backendClasses[$backendname])) {
                    $backendname = $this->backendClasses[$backendname];
                }
                $backend = new $backendname($matrixOperator,$origName);
            } else {
                $backend = new RindowBlasBackend($matrixOperator);
            }
        }
        $this->backend = $backend;
        $this->matrixOperator = $matrixOperator;
    }

    public function __get( string $name ) : object
    {
        if(!method_exists($this,$name)) {
            throw new LogicException('Unknown builder: '.$name);
        }
        return $this->$name();
    }

    public function __set( string $name, mixed $value ) : void
    {
        throw new LogicException('Invalid operation to set');
    }

    public function backend() : object
    {
        return $this->backend;
    }

    public function deviceType() : string
    {
        return $this->backend->deviceType();
    }

    public function models() : object
    {
        if($this->models==null) {
            $this->models = new Models($this);
        }
        return $this->models;
    }

    public function layers() : object
    {
        if($this->layers==null) {
            $this->layers = new Layers($this->backend);
        }
        return $this->layers;
    }

    public function losses() : object
    {
        if($this->losses==null) {
            $this->losses = new Losses($this->backend);
        }
        return $this->losses;
    }

    public function metrics() : object
    {
        if($this->metrics==null) {
            $this->metrics = new Metrics($this->backend);
        }
        return $this->metrics;
    }

    public function optimizers() : object
    {
        if($this->optimizers==null) {
            $this->optimizers = new Optimizers($this->backend);
        }
        return $this->optimizers;
    }

    public function datasets() : object
    {
        if($this->datasets==null) {
            $this->datasets = new Datasets($this->matrixOperator);
        }
        return $this->datasets;
    }

    public function data() : object
    {
        if($this->data==null) {
            $this->data = new Data($this->matrixOperator);
        }
        return $this->data;
    }

    public function utils() : object
    {
        if($this->utils==null) {
            $this->utils = new Utils($this->matrixOperator);
        }
        return $this->utils;
    }

    public function gradient() : object
    {
        if($this->gradient==null) {
            $this->gradient = new Gradient($this->backend);
        }
        return $this->gradient;
    }

    public function with(mixed ...$args) : mixed
    {
        return Execute::with(...$args);
    }
}
