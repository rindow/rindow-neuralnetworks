<?php
namespace Rindow\NeuralNetworks\Layer;

class GlobalAveragePooling1D extends AbstractGlobalAveragePooling
{
    protected int $rank = 1;
    protected string $defaultLayerName = 'globalaveragepooling1d';
}
