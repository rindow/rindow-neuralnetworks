<?php
namespace Rindow\NeuralNetworks\Layer;

class GlobalAveragePooling3D extends AbstractGlobalAveragePooling
{
    protected $rank = 3;
    protected $defaultLayerName = 'globalaveragepooling3d';
}
