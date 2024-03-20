<?php
namespace Rindow\NeuralNetworks\Builder;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Variable as VariableInterface;
use Rindow\NeuralNetworks\Gradient\Core\Variable;
use Rindow\NeuralNetworks\Gradient\Core\Modules;
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
use Rindow\NeuralNetworks\Gradient\Func\Equal;
use Rindow\NeuralNetworks\Gradient\Func\NotEqual;
use Rindow\NeuralNetworks\Gradient\Func\Cast;
use Rindow\NeuralNetworks\Gradient\Func\ZerosLike;
use Rindow\NeuralNetworks\Gradient\Func\Reshape;
use Rindow\NeuralNetworks\Gradient\Func\Transpose;
use Rindow\NeuralNetworks\Gradient\Func\Shape;
use Rindow\NeuralNetworks\Gradient\Func\Get;
use Rindow\NeuralNetworks\Gradient\Func\Scale;
use Rindow\NeuralNetworks\Gradient\Func\Zeros;
use Rindow\NeuralNetworks\Gradient\Func\Ones;
use Rindow\NeuralNetworks\Gradient\Func\Bandpart;
use Rindow\NeuralNetworks\Gradient\Func\Increment;
use Rindow\NeuralNetworks\Gradient\Func\Greater;
use Rindow\NeuralNetworks\Gradient\Func\Less;
use Rindow\NeuralNetworks\Gradient\Func\Repeat;

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

    public function modules($modules=null)
    {
        return new Modules($modules);
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
        int $keepdims=null,
    )
    {
        $func = new ReduceMean(
            $this->backend,
            axis:$axis,
            keepdims:$keepdims,
        );
        return $func($x);
    }

    public function reduceSum(
        $x,
        int $axis=null,
        int $keepdims=null,
    )
    {
        $func = new ReduceSum(
            $this->backend,
            axis:$axis,
            keepdims:$keepdims,
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

    public function equal($x,$y)
    {
        $func = new Equal($this->backend);
        return $func($x,$y);
    }

    public function notEqual($x,$y)
    {
        $func = new NotEqual($this->backend);
        return $func($x,$y);
    }

    public function cast($x,$dtype)
    {
        $func = new Cast($this->backend,$dtype);
        return $func($x);
    }

    public function zerosLike($x)
    {
        $func = new ZerosLike($this->backend);
        return $func($x);
    }

    public function reshape($x, $shape)
    {
        $func = new Reshape($this->backend);
        return $func($x,$shape);
    }

    public function transpose(
        $x,
        array $perm=null,
    )
    {
        $func = new Transpose(
            $this->backend,
            $perm,
        );
        return $func($x);
    }

    public function shape(
        $x,
    )
    {
        $func = new Shape($this->backend);
        return $func($x);
    }

    public function get(
        $x,
        $offset,
        $count=null,
    )
    {
        if($count===null) {
            $count = 0;
        }
        $func = new Get($this->backend);
        return $func($x,$offset,$count);
    }

    public function scale(
        $alpha,
        $x,
    )
    {
        $func = new Scale($this->backend);
        return $func($alpha,$x);
    }

    public function zeros(
        $shape,
        $dtype=null,
    )
    {
        $func = new Zeros($this->backend,$dtype);
        return $func($shape);
    }

    public function ones(
        $shape,
        $dtype=null,
    )
    {
        $func = new Ones($this->backend,$dtype);
        return $func($shape);
    }

    public function bandpart(
        $x,
        $lower,
        $upper,
    )
    {
        $func = new Bandpart($this->backend,$lower,$upper);
        return $func($x);
    }

    public function increment(
        $x,
        $beta,
        $alpha=null,
    )
    {
        $alpha = $alpha ?? 1.0;
        $func = new Increment($this->backend);
        return $func($x,$beta,$alpha);
    }

    public function greater(
        $x,
        $alpha,
    )
    {
        $func = new Greater($this->backend);
        return $func($x,$alpha);
    }

    public function less(
        $x,
        $alpha,
    )
    {
        $func = new Less($this->backend);
        return $func($x,$alpha);
    }

    public function repeat(
        $x,
        $repeats,
        $axis=null,
        $keepdims=null,
    )
    {
        $func = new Repeat($this->backend,axis:$axis,keepdims:$keepdims);
        return $func($x,$repeats);
    }

}
