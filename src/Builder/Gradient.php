<?php
namespace Rindow\NeuralNetworks\Builder;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\Variable;
use Rindow\NeuralNetworks\Gradient\Core\Undetermined;
use Rindow\NeuralNetworks\Gradient\Core\UndeterminedNDArray;
use Rindow\NeuralNetworks\Gradient\Core\GradientTape;
use Rindow\NeuralNetworks\Gradient\Func\Square;
use Rindow\NeuralNetworks\Gradient\Func\Sqrt;
use Rindow\NeuralNetworks\Gradient\Func\Exp;
use Rindow\NeuralNetworks\Gradient\Func\Log;
use Rindow\NeuralNetworks\Gradient\Func\Add;
use Rindow\NeuralNetworks\Gradient\Func\Sub;
use Rindow\NeuralNetworks\Gradient\Func\Mul;
use Rindow\NeuralNetworks\Gradient\Func\Div;
use Rindow\NeuralNetworks\Gradient\Func\Matmul;

class Gradient
{
    protected $backend;

    public function __construct($backend)
    {
        $this->backend = $backend;
    }

    public function Variable(NDArray $variable,array $options=null)
    {
        return new Variable($this->backend,$variable,$options);
    }

    public function GradientTape($persistent=null)
    {
        return new GradientTape($this->backend,$persistent);
    }

    public function isUndetermined($variable) : bool
    {
        if(($variable instanceof Undetermined)||
            ($variable instanceof UndeterminedNDArray)) {
            return true;
        } else {
            return false;
        }
    }

    public function Undetermined()
    {
        return new Undetermined();
    }

    public function UndeterminedNDArray()
    {
        return new UndeterminedNDArray();
    }

    public function square($x)
    {
        $func = new Square($this->backend);
        return $func($x);
    }

    public function sqrt($x)
    {
        $func = new Sqrt($this->backend);
        return $func($x);
    }

    public function exp($x)
    {
        $func = new Exp($this->backend);
        return $func($x);
    }

    public function log($x)
    {
        $func = new Log($this->backend);
        return $func($x);
    }

    public function add($x,$y)
    {
        $func = new Add($this->backend);
        return $func($x,$y);
    }

    public function sub($x,$y)
    {
        $func = new Sub($this->backend);
        return $func($x,$y);
    }

    public function mul($x,$y)
    {
        $func = new Mul($this->backend);
        return $func($x,$y);
    }

    public function div($x,$y)
    {
        $func = new Div($this->backend);
        return $func($x,$y);
    }

    public function matmul($x,$y,array $options=null)
    {
        $func = new Matmul($this->backend,$options);
        return $func($x,$y);
    }

}
