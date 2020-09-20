<?php
namespace Rindow\NeuralNetworks\Activation;

use InvalidArgumentException;

class FunctionFactory
{
    static $functions = [
        'sigmoid' => Sigmoid::class,
        'softmax' => Softmax::class,
        'relu' => ReLU::class,
        'tanh' => Tanh::class,
    ];
    static public function factory($backend, string $name) : Activation
    {
        if(isset(self::$functions[$name])) {
            $class = self::$functions[$name];
            return new $class($backend);
        }
        if(is_subclass_of($name, Activation::class)) {
            return new $class($backend);
        }
        throw new InvalidArgumentException('invalid function name');
    }
}
