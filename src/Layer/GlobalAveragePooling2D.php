<?php
namespace Rindow\NeuralNetworks\Layer;

class GlobalAveragePooling2D extends AbstractGlobalAveragePooling
{
    protected int $rank = 2;
    protected string $defaultLayerName = 'globalaveragepooling2d';
}
