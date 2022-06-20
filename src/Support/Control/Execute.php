<?php
namespace Rindow\NeuralNetworks\Support\Control;

use Throwable;

class Execute
{
    static public function with(Context $context, callable $func)
    {
        $context->enter();
        try {
            $result = $func($context);
        } catch(Throwable $e) {
            if($context->exit($e))
                return null;
            throw $e;
        }
        $context->exit(null);
        return $result;
    }
}
