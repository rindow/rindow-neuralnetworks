<?php
namespace Rindow\NeuralNetworks\Layer;

class GlobalMaxPooling2D extends AbstractGlobalMaxPooling
{
    protected int $rank = 2;
    protected string $defaultLayerName = 'globalmaxpooling2d';
}
