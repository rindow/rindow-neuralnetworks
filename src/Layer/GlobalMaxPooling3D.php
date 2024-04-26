<?php
namespace Rindow\NeuralNetworks\Layer;

class GlobalMaxPooling3D extends AbstractGlobalMaxPooling
{
    protected int $rank = 3;
    protected string $defaultLayerName = 'globalmaxpooling3d';
}
