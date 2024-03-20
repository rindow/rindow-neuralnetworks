<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use InvalidArgumentException;
use LogicException;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;
use Rindow\NeuralNetworks\Gradient\Core\Scalar;
use Rindow\NeuralNetworks\Gradient\Core\ArrayShape;
use Rindow\NeuralNetworks\Gradient\ArrayShape as ArrayShapeInterface;
use Rindow\NeuralNetworks\Gradient\Scalar as ScalarInterface;
use Interop\Polite\Math\Matrix\NDArray;
use function Rindow\Math\Matrix\R;

class Get extends AbstractFunction
{
    protected $numOfInputs = 3;
    protected $outputs;

    protected function preprocess(array $inputs) : array
    {
        if(is_numeric($inputs[1])) {
            $inputs[1] = new Scalar($inputs[1]);
        }
        if(is_numeric($inputs[2])) {
            $inputs[2] = new Scalar($inputs[2]);
        }
        return $inputs;
    }

    protected function call(array $inputs) : array
    {
        $K = $this->backend;
        $array  = $inputs[0];
        $offset = $inputs[1];
        $count = $inputs[2];
        if($offset instanceof ScalarInterface) {
            $offset = $offset->value();
        }
        if($count instanceof ScalarInterface) {
            $count = $count->value();
        }

        if($array instanceof ArrayShapeInterface) {
            $value = $this->getArrayShape($array,$offset,$count);
        } elseif($array instanceof NDArray) {
            $value = $this->getNDArray($array,$offset,$count);
        } else {
            if(is_object($array)) {
                $type = get_class($array);
            } else {
                $type = gettype($array);
            }
            throw new InvalidArgumentException("arg #1 is invalid data type.: ".$type);
        }

        $this->unbackpropagatables = [true];
        return [$value];
    }

    protected function getArrayShape($array,$offset,$count)
    {
        $K = $this->backend;
        if($count===0) {
            $value = new Scalar($array[$offset]);
            return $value;
        }
        
        if($count<0) {
            $count = count($array) + $count + 1;
        }
        $value = [];
        foreach ($array as $i => $v) {
            if($i>=$offset && $i<$offset+$count) {
                $value[] = $v;
            }
        }
        $value = new ArrayShape($value);
        return $value;
    }

    protected function getNDArray($array,$offset,$count)
    {
        $K = $this->backend;
        if($count===0) {
            if($array->ndim()==0) {
                throw new InvalidArgumentException("arg #1 must not be scalar.");
            } elseif($array->ndim()==1) {
                $value = $array[R($offset,$offset+1)];
                $value = $K->ndarray($value->reshape([]));
                $value = new Scalar($value);
            } else {
                $value = $array[$offset];
            }
            return $value;
        }

        if($count<0) {
            $count = count($array) + $count + 1;
        }

        $value = $array[R($offset,$offset+$count)];
        return $value;
    }

    protected function differentiate(array $dOutputs) : array
    {
        throw new LogicException('This function is not differentiable');
    }
}
