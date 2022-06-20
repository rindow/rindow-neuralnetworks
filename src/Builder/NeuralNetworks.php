<?php
namespace Rindow\NeuralNetworks\Builder;

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend as RindowBlasBackend;
use Rindow\NeuralNetworks\Backend\RindowCLBlast\Backend as RindowCLBlastBackend;
use Rindow\NeuralNetworks\Support\Control\Execute;
use Rindow\NeuralNetworks\Support\Control\Context;
use LogicException;

class NeuralNetworks
{
    protected $backendClasses = [
        'rindowblas' => RindowBlasBackend::class,
        'rindowclblast' => RindowCLBlastBackend::class,
    ];
    protected $backend;
    protected $matrixOperator;
    protected $functions;
    protected $models;
    protected $layers;
    protected $losses;
    protected $optimizers;
    protected $networks;
    protected $datasets;
    protected $data;
    protected $utils;
    protected $gradient;

    public function __construct(object $matrixOperator=null, object $backend=null)
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

    public function __get( string $name )
    {
        if(!method_exists($this,$name)) {
            throw new LogicException('Unknown builder: '.$name);
        }
        return $this->$name();
    }

    public function __set( string $name, $value ) : void
    {
        throw new LogicException('Invalid operation to set');
    }

    public function backend()
    {
        return $this->backend;
    }

    public function models()
    {
        if($this->models==null) {
            $this->models = new Models($this->backend,$this);
        }
        return $this->models;
    }

    public function layers()
    {
        if($this->layers==null) {
            $this->layers = new Layers($this->backend);
        }
        return $this->layers;
    }

    public function losses()
    {
        if($this->losses==null) {
            $this->losses = new Losses($this->backend);
        }
        return $this->losses;
    }

    public function optimizers()
    {
        if($this->optimizers==null) {
            $this->optimizers = new Optimizers($this->backend);
        }
        return $this->optimizers;
    }

    public function datasets()
    {
        if($this->datasets==null) {
            $this->datasets = new Datasets($this->matrixOperator);
        }
        return $this->datasets;
    }

    public function data()
    {
        if($this->data==null) {
            $this->data = new Data($this->matrixOperator);
        }
        return $this->data;
    }

    public function utils()
    {
        if($this->utils==null) {
            $this->utils = new Utils($this->matrixOperator);
        }
        return $this->utils;
    }

    public function gradient()
    {
        if($this->gradient==null) {
            $this->gradient = new Gradient($this->backend);
        }
        return $this->gradient;
    }

    public function with(...$args)
    {
        return Execute::with(...$args);
    }
}
