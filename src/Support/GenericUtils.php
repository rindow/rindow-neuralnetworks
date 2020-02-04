<?php
namespace Rindow\NeuralNetworks\Support;

use InvalidArgumentException;

trait GenericUtils
{
    protected function extractArgs(array $keywords,array $kwargs=null)
    {
        if($kwargs===null) {
            $kwargs = [];
        }
        $vars = [];
        foreach ($kwargs as $key => $value) {
            if(array_key_exists($key, $keywords)) {
                $vars[$key] = $value;
            } else {
                throw new InvalidArgumentException(
                    'Unexpected keyword argument "'.$key.'"');
            }
        }
        foreach ($keywords as $key => $default) {
            if(!array_key_exists($key,$kwargs)) {
                $vars[$key] = $default;
            }
        }
        return $vars;
    }

}
