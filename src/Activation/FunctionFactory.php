<?php
namespace Rindow\NeuralNetworks\Activation;

use InvalidArgumentException;

class FunctionFactory
{
    /** @var array<string,string|null> $functions */
    static array $functions = [
        'sigmoid' => Sigmoid::class,
        'softmax' => Softmax::class,
        'relu' => ReLU::class,
        'tanh' => Tanh::class,
        'linear' => null,
    ];
    static public function factory(object $backend, string $name) : ?Activation
    {
        if(array_key_exists($name,self::$functions)) {
            $class = self::$functions[$name];
            if($class==null) {
                return null;
            }
            return new $class($backend);
        }
        if(is_subclass_of($name, Activation::class)) {
            return new $name($backend);
        }
        
        throw new InvalidArgumentException('invalid function name:'.$name);
    }
}
