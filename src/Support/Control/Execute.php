<?php
namespace Rindow\NeuralNetworks\Support\Control;

use Throwable;

class Execute
{
    /**
     * @param array<mixed,mixed> $args
     */
    static public function with(
        Context $context,
        callable $func=null,
        array $args=null,
        bool $without_ctx=null,
        ) : mixed
    {
        if($args===null) {
            $args = [];
        }
        $context->enter();
        try {
            if($without_ctx) {
                $result = $func(...$args);
            } else {
                $result = $func($context, ...$args);
            }
        } catch(Throwable $e) {
            if(!$context->exit($e)) {
                throw $e;
            }
            $result = null;
        }
        $context->exit(null);
        return $result;
    }
}
