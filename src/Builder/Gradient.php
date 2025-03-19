<?php
namespace Rindow\NeuralNetworks\Builder;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Variable as VariableIF;
use Rindow\NeuralNetworks\Gradient\MaskedNDArray;
use Rindow\NeuralNetworks\Gradient\Module;
use Rindow\NeuralNetworks\Gradient\ArrayShape;
use Rindow\NeuralNetworks\Gradient\ArraySpec;
use Rindow\NeuralNetworks\Gradient\Core\ArraySpec as ArraySpecImpl;
use Rindow\NeuralNetworks\Gradient\Core\Variable as VariableImpl;
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
use Rindow\NeuralNetworks\Gradient\Func\Reshape;
use Rindow\NeuralNetworks\Gradient\Func\Transpose;
use Rindow\NeuralNetworks\Gradient\Func\Shape;
use Rindow\NeuralNetworks\Gradient\Func\Get;
use Rindow\NeuralNetworks\Gradient\Func\Scale;
use Rindow\NeuralNetworks\Gradient\Func\Zeros;
use Rindow\NeuralNetworks\Gradient\Func\ZerosLike;
use Rindow\NeuralNetworks\Gradient\Func\Ones;
use Rindow\NeuralNetworks\Gradient\Func\OnesLike;
use Rindow\NeuralNetworks\Gradient\Func\Bandpart;
use Rindow\NeuralNetworks\Gradient\Func\Increment;
use Rindow\NeuralNetworks\Gradient\Func\Greater;
use Rindow\NeuralNetworks\Gradient\Func\Less;
use Rindow\NeuralNetworks\Gradient\Func\Repeat;
use Rindow\NeuralNetworks\Gradient\Func\Split;
use Rindow\NeuralNetworks\Gradient\Func\Nop;

class Gradient
{
    protected object $backend;

    public function __construct(object $backend)
    {
        $this->backend = $backend;
    }

    public function constant(mixed $value, ?int $dtype=null) : NDArray
    {
        return $this->backend->array($value, dtype:$dtype);
    }

    public function Variable(mixed $variable, mixed ...$options) : VariableIF
    {
        if(GraphFunction::$mode==GraphFunction::EXECUTING) {
            return $variable;
        }
        return new VariableImpl($this->backend,$variable,...$options);
    }

    public function ndarray(NDArray $value) : NDArray
    {
        if($value instanceof VariableIF) {
            $value = $value->value();
        }
        if($value instanceof MaskedNDArray) {
            $value = $value->value();
        }
        return $value;
    }

    /**
     * @param array<mixed> $values
     * @return array<VariableIF>
     */
    public function toVariables(array $values, mixed ...$options) : array
    {
        if(GraphFunction::$mode==GraphFunction::EXECUTING) {
            return $values;
        }
        $variables = [];
        foreach($values as $value) {
            $variables[] = new VariableImpl($this->backend,$value,...$options);
        }
        return $variables;
    }

    /**
     * @param ArrayShape|array<int> $shape
     */
    public function ArraySpec(
        ArrayShape|array $shape,
        ?int $dtype=null,
        ?string $name=null,
    ) : ArraySpec
    {
        return new ArraySpecImpl($shape, dtype:$dtype, name:$name);
    }

    /**
     * @param array<Module> $modules
     */
    public function modules(?array $modules=null) : object
    {
        return new Modules($modules);
    }

    public function GradientTape(?bool $persistent=null, ?string $name=null) : object
    {
        return new GradientTape($this->backend, persistent:$persistent, name:$name);
    }

    public function Function(callable $func, mixed ...$options) : object
    {
        return new GraphFunction($this->backend, $func, ...$options);
    }

    public function isUndetermined(mixed $variable) : bool
    {
        if($variable instanceof VariableIF) {
            /** @var VariableImpl $variable */
            if($variable->isUndetermined()) {
                return true;
            }
        }
        return false;
    }

    public function stopGradient(NDArray $variable, ?string $name=null) : NDArray
    {
        $func = new StopGradient($this->backend, name:$name);
        return $func($variable);
    }

    public function square(NDArray $x, ?string $name=null) : NDArray
    {
        $func = new Square($this->backend, name:$name);
        return $func($x);
    }

    public function sqrt(NDArray $x, ?string $name=null) : NDArray
    {
        $func = new Sqrt($this->backend, name:$name);
        return $func($x);
    }

    public function exp(NDArray $x, ?string $name=null) : NDArray
    {
        $func = new Exp($this->backend, name:$name);
        return $func($x);
    }

    public function log(NDArray $x, ?string $name=null) : NDArray
    {
        $func = new Log($this->backend, name:$name);
        return $func($x);
    }

    public function add(NDArray $x, NDArray $y, ?string $name=null) : NDArray
    {
        $func = new Add($this->backend, name:$name);
        return $func($x,$y);
    }

    public function sub(NDArray $x, NDArray $y, ?string $name=null) : NDArray
    {
        $func = new Sub($this->backend, name:$name);
        return $func($x,$y);
    }

    public function mul(NDArray $x, NDArray $y, ?string $name=null) : NDArray
    {
        $func = new Mul($this->backend, name:$name);
        return $func($x,$y);
    }

    public function div(NDArray $x, NDArray $y, ?string $name=null) : NDArray
    {
        $func = new Div($this->backend, name:$name);
        return $func($x,$y);
    }

    public function matmul(
        NDArray $x, NDArray $y,
        ?bool $transpose_a=null,
        ?bool $transpose_b=null,
        ?string $name=null,
    ) : NDArray
    {
        $func = new Matmul(
            $this->backend,
            transpose_a:$transpose_a,
            transpose_b:$transpose_b,
            name:$name,
        );
        return $func($x,$y);
    }

    public function reduceMean(
        NDArray $x,
        ?int $axis=null,
        ?bool $keepdims=null,
        ?string $name=null,
    ) : NDArray
    {
        $func = new ReduceMean(
            $this->backend,
            axis:$axis,
            keepdims:$keepdims,
            name:$name,
        );
        return $func($x);
    }

    public function reduceSum(
        NDArray $x,
        ?int $axis=null,
        ?bool $keepdims=null,
        ?string $name=null,
    ) : NDArray
    {
        $func = new ReduceSum(
            $this->backend,
            axis:$axis,
            keepdims:$keepdims,
            name:$name,
        );
        return $func($x);
    }

    public function clipByValue(
        NDArray $x,
        float $min,
        float $max,
        ?string $name=null,
    ) : NDArray
    {
        $func = new ClipByValue(
            $this->backend,
            $min,
            $max,
            name:$name,
        );
        return $func($x);
    }

    public function equal(NDArray $x, NDArray $y, ?string $name=null) : NDArray
    {
        $func = new Equal($this->backend, name:$name);
        return $func($x,$y);
    }

    public function notEqual(NDArray $x, NDArray $y, ?string $name=null) : NDArray
    {
        $func = new NotEqual($this->backend, name:$name);
        return $func($x,$y);
    }

    public function cast(NDArray $x, int $dtype, ?string $name=null) : NDArray
    {
        $func = new Cast($this->backend, dtype:$dtype, name:$name);
        return $func($x);
    }

    public function zerosLike(NDArray $x, ?string $name=null) : NDArray 
    {
        $func = new ZerosLike($this->backend, name:$name);
        return $func($x);
    }

    public function onesLike(NDArray $x, ?string $name=null) : NDArray 
    {
        $func = new OnesLike($this->backend, name:$name);
        return $func($x);
    }

    public function reshape(NDArray $x, mixed $shape, ?string $name=null) : NDArray
    {
        $func = new Reshape($this->backend, name:$name);
        return $func($x,$shape);
    }

    /**
     * @param array<int> $perm
     */
    public function transpose(
        NDArray $x,
        ?array $perm=null,
        ?string $name=null,
    ) : NDArray
    {
        $func = new Transpose(
            $this->backend,
            $perm,
            name:$name,
        );
        return $func($x);
    }

    public function shape(
        mixed $x,
        ?string $name=null,
    ) : NDArray
    {
        $func = new Shape($this->backend, name:$name);
        return $func($x);
    }

    public function get(
        mixed $x,
        mixed $offset,
        mixed $count=null,
        ?string $name=null,
    ) : mixed
    {
        if($count===null) {
            $count = 0;
        }
        $func = new Get($this->backend, name:$name);
        return $func($x,$offset,$count);
    }

    public function scale(
        mixed $alpha,
        NDArray $x,
        ?string $name=null,
    ) : NDArray
    {
        $func = new Scale($this->backend, name:$name);
        return $func($alpha,$x);
    }

    public function zeros(
        mixed $shape,
        ?int $dtype=null,
        ?string $name=null,
    ) : NDArray
    {
        $func = new Zeros($this->backend, dtype:$dtype, name:$name);
        return $func($shape);
    }

    public function ones(
        mixed $shape,
        ?int $dtype=null,
        ?string $name=null,
    ) : NDArray
    {
        $func = new Ones($this->backend, dtype:$dtype, name:$name);
        return $func($shape);
    }

    public function bandpart(
        NDArray $x,
        int $lower,
        int $upper,
        ?string $name=null,
    ) : NDArray
    {
        $func = new Bandpart($this->backend,$lower,$upper, name:$name);
        return $func($x);
    }

    public function increment(
        NDArray $x,
        mixed $beta,
        mixed $alpha=null,
        ?string $name=null,
    ) : NDArray
    {
        $alpha = $alpha ?? 1.0;
        $func = new Increment($this->backend, name:$name);
        return $func($x,$beta,$alpha);
    }

    public function greater(
        NDArray $x,
        mixed $alpha,
        ?string $name=null,
    ) : NDArray
    {
        $func = new Greater($this->backend, name:$name);
        return $func($x,$alpha);
    }

    public function less(
        NDArray $x,
        mixed $alpha,
        ?string $name=null,
    ) : NDArray
    {
        $func = new Less($this->backend, name:$name);
        return $func($x,$alpha);
    }

    public function repeat(
        NDArray $x,
        mixed $repeats,
        ?int $axis=null,
        ?bool $keepdims=null,
        ?string $name=null,
    ) : NDArray
    {
        $func = new Repeat($this->backend,axis:$axis,keepdims:$keepdims, name:$name);
        return $func($x,$repeats);
    }

    /**
     * @param array<int> $sizeSplits
     * @return array<NDArray>
     */
    public function split(
        NDArray $x,
        array $sizeSplits,
        ?int $axis=null,
        ?string $name=null,
    ) : array
    {
        $func = new Split($this->backend,$sizeSplits,axis:$axis, name:$name);
        return $func($x);
    }
    
    public function nop(
        NDArray $x,
        mixed ...$options,
    ) : NDArray
    {
        $func = new Nop($this->backend,...$options);
        return $func($x);
    }
}
