<?php
namespace Rindow\NeuralNetworks\Support\Control;

use Throwable;

interface Context
{
    public function enter() : void;
    public function exit(?Throwable $e=null) : bool;
}
