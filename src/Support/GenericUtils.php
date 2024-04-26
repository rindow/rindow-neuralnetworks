<?php
namespace Rindow\NeuralNetworks\Support;

use InvalidArgumentException;

trait GenericUtils
{
    public static int $nameNumbering = 0;

    protected function initName(?string $name, ?string $default) : void
    {
        if($name===null) {
            if(self::$nameNumbering==0) {
                $name = $default;
            } else {
                $name = $default.'_'.self::$nameNumbering;
            }
            self::$nameNumbering++;
        }
        $this->name = $name;
    }
}
