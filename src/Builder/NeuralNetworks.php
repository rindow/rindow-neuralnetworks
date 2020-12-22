<?php
namespace Rindow\NeuralNetworks\Builder;

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend as RindowBlasBackend;
use Rindow\NeuralNetworks\Backend\RindowCLBlast\Backend as RindowCLBlastBackend;

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
    protected $utils;

    public function __construct($matrixOperator=null,$backend=null)
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

    public function utils()
    {
        if($this->utils==null) {
            $this->utils = new Utils($this->matrixOperator);
        }
        return $this->utils;
    }
}
