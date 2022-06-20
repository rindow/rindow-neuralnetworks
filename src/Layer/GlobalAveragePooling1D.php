<?php
namespace Rindow\NeuralNetworks\Layer;

class GlobalAveragePooling1D extends AbstractGlobalAveragePooling
{
    protected $rank = 1;
    protected $defaultLayerName = 'globalaveragepooling1d';
}
