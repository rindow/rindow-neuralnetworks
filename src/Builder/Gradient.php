<?php
namespace Rindow\NeuralNetworks\Builder;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Variable as VariableInterface;
use Rindow\NeuralNetworks\Gradient\Core\Variable;
use Rindow\NeuralNetworks\Gradient\Core\GradientTape;
use Rindow\NeuralNetworks\Gradient\Core\GraphFunction;
use Rindow\NeuralNetworks\Gradient\Core\StopGradient;
use Rindow\NeuralNetworks\Gradient\Func\Square;
use Rindow\NeuralNetworks\Gradient\Func\Sqrt;
use Rindow\NeuralNetworks\Gradient\Func\Exp;
use Rindow\NeuralNetworks\Gradient\Func\Log;
use Rindow\NeuralNetworks\Gradient\Func\Add;
use Rindow\NeuralNetworks\Gradient\Func\Sub;
use Rindow\NeuralNetworks\Gradient\Func\Mul;
use Rindow\NeuralNetworks\Gradient\Func\Div;
use Rindow\NeuralNetworks\Gradient\Func\Matmul;
use Rindow\NeuralNetworks\Gradient\Func\ReduceMean;
use Rindow\NeuralNetworks\Gradient\Func\ReduceSum;
use Rindow\NeuralNetworks\Gradient\Func\ClipByValue;

class Gradient
{
    protected $backend;

    public function __construct(object $backend)
    {
        $this->backend = $backend;
    }

    public function Variable($variable, ...$options)
    {
        if(GraphFunction::$mode==GraphFunction::EXECUTING) {
            return $variable;
        }
        return new Variable($this->backend,$variable,...$options);
    }

    public function toVariables(array $values, ...$options) : array
    {
        if(GraphFunction::$mode==GraphFunction::EXECUTING) {
            return $values;
        }
        $variables = [];
        foreach($values as $value) {
            $variables[] = new Variable($this->backend,$value,...$options);
        }
        return $variables;
    }

    public function GradientTape($persistent=null)
    {
        return new GradientTape($this->backend,$persistent);
    }

    public function Function(callable $func, ...$options)
    {
        return new GraphFunction($this->backend, $func, ...$options);
    }

    public function isUndetermined($variable) : bool
    {
        if($variable instanceof VariableInterface) {
            if($variable->isUndetermined()) {
                return true;
            }
        }
        return false;
    }

    public function stopGradient($variable)
    {
        $func = new StopGradient($this->backend);
        return $func($variable);
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

    public function matmul(
        $x,$y,
        bool $transpose_a=null,
        bool $transpose_b=null,
    )
    {
        $func = new Matmul(
            $this->backend,
            transpose_a:$transpose_a,
            transpose_b:$transpose_b,
        );
        return $func($x,$y);
    }

    public function reduceMean(
        $x,
        int $axis=null,
    )
    {
        $func = new ReduceMean(
            $this->backend,
            axis:$axis,
        );
        return $func($x);
    }

    public function reduceSum(
        $x,
        int $axis=null,
    )
    {
        $func = new ReduceSum(
            $this->backend,
            axis:$axis,
        );
        return $func($x);
    }

    public function clipByValue(
        $x,
        float $min,
        float $max,
    )
    {
        $func = new ClipByValue(
            $this->backend,
            $min,
            $max,
        );
        return $func($x);
    }

}
