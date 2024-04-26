<?php
namespace Rindow\NeuralNetworks\Layer;

class GlobalAveragePooling3D extends AbstractGlobalAveragePooling
{
    protected int $rank = 3;
    protected string $defaultLayerName = 'globalaveragepooling3d';
}
