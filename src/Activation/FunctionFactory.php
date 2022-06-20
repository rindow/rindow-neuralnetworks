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
        'linear' => null,
    ];
    static public function factory(object $backend, string $name) : ?Activation
    {
        if(array_key_exists($name,self::$functions)) {
            $class = self::$functions[$name];
            if($class===null) {
                return null;
            }
            return new $class($backend);
        }
        if(is_subclass_of($name, Activation::class)) {
            return new $class($backend);
        }
        
        if(is_string($name)||is_numeric($name)) {
            ;
        } elseif(is_object($name)) {
            $name = 'object:'.get_class($name);
        } else {
            $name = gettype($name);
        }
        throw new InvalidArgumentException('invalid function name:'.$name);
    }
}
