<?php
namespace Rindow\NeuralNetworks\Callback;

use Rindow\NeuralNetworks\Model\Model;

/**
 *
 */
interface Callback extends Events
{
    public function setModel(Model $model) : void;
    public function getModel() : ?Model;
}
