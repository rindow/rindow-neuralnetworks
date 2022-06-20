<?php
namespace Rindow\NeuralNetworks\Layer;

class GlobalMaxPooling3D extends AbstractGlobalMaxPooling
{
    protected $rank = 3;
    protected $defaultLayerName = 'globalmaxpooling3d';
}
