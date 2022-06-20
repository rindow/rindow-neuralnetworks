<?php
namespace Rindow\NeuralNetworks\Layer;

class GlobalAveragePooling2D extends AbstractGlobalAveragePooling
{
    protected $rank = 2;
    protected $defaultLayerName = 'globalaveragepooling2d';
}
